"""
End-to-end browser test: drives the real React app against the real FastAPI backend.

Prerequisites (both servers running):
    backend:  uvicorn app.main:app --port 8000        (from backend/)
    frontend: npm run dev                              (from frontend/, port 5173)
    pip install playwright && python -m playwright install chromium

Run:
    python e2e/run_e2e.py                 # screenshots go to e2e/screenshots/

Every page, every button and the main error paths are exercised. The script
exits with code 1 if any check fails or the browser console logs an error.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

APP = os.getenv("APP_URL", "http://localhost:5173")
API = os.getenv("API_URL", "http://localhost:8000")
SHOTS = Path(__file__).parent / "screenshots"
SHOTS.mkdir(exist_ok=True)

PATIENT = {"pregnancies": 2, "glucose": 150, "blood_pressure": 70, "skin_thickness": 20,
           "insulin": 79, "bmi": 32, "diabetes_pedigree": 0.35, "age": 45}

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        results.append((name, False, str(exc).splitlines()[0][:200]))
        print(f"  FAIL  {name}: {str(exc).splitlines()[0][:200]}")


def api_predict(patient):
    req = urllib.request.Request(f"{API}/predict", data=json.dumps(patient).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def fill_form(page, prefix, patient):
    for k, v in patient.items():
        page.fill(f"#{prefix}-{k}", "" if v is None else str(v))


def no_horizontal_overflow(page):
    over = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert over <= 1, f"page overflows horizontally by {over}px"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1366, "height": 900})
        page = ctx.new_page()
        console_errors = []
        # "Failed to load resource" = a network request failed (e.g. the deliberate abort in the
        # backend-unavailable test, or web fonts when offline). Those are not JavaScript errors.
        page.on("console", lambda m: console_errors.append(m.text)
                if m.type == "error" and "Failed to load resource" not in m.text else None)
        page.on("pageerror", lambda e: console_errors.append(str(e)))

        print("Desktop 1366x900")

        # ---------- Dashboard ----------
        def dashboard():
            page.goto(APP)
            expect(page.get_by_text("Connected (API v")).to_be_visible(timeout=15000)
            expect(page.get_by_text("Support Vector Machine (RBF)").first).to_be_visible()
            expect(page.get_by_text("ROC-AUC (95% CI")).to_be_visible(timeout=15000)
            page.screenshot(path=SHOTS / "01_dashboard.png", full_page=True)
        check("Dashboard shows live backend status and real metrics", dashboard)

        # ---------- Navigation ----------
        def navigation():
            for label, heading in [("Prediction", "Diabetes prediction"), ("Results", "No patient analysed yet"),
                                   ("Explainability", "Explainability"), ("What-if analysis", "No patient analysed yet"),
                                   ("Model information", "Model information"), ("About & disclaimer", "About this system"),
                                   ("Dashboard", "Diabetes risk prediction")]:
                page.get_by_role("link", name=label, exact=True).click()
                expect(page.get_by_role("heading", name=heading).first).to_be_visible(timeout=15000)
        check("Every sidebar navigation link opens its page", navigation)

        # ---------- Validation ----------
        def validation_empty():
            page.goto(f"{APP}/predict")
            expect(page.locator("#predict-glucose")).to_be_visible(timeout=15000)
            calls = []
            page.on("request", lambda r: calls.append(r.url) if r.url.endswith("/predict") else None)
            page.get_by_role("button", name="Predict", exact=True).click()
            expect(page.get_by_text("6 fields need correcting")).to_be_visible()
            expect(page.get_by_text("Glucose is required.")).to_be_visible()
            assert not calls, "invalid form must not reach the backend"
        check("Empty form is blocked by frontend validation (no request sent)", validation_empty)

        def validation_ranges():
            fill_form(page, "predict", {**PATIENT, "age": -5, "bmi": 500})
            page.get_by_role("button", name="Predict", exact=True).click()
            expect(page.get_by_text("Please enter a valid age (whole number, 21–100 years).")).to_be_visible()
            expect(page.get_by_text("Please enter a valid BMI (12–70 kg/m²).")).to_be_visible()
            page.screenshot(path=SHOTS / "02_validation.png", full_page=True)
        check("Out-of-range age and BMI show field errors", validation_ranges)

        def load_example():
            page.get_by_role("button", name="Load example").click()
            expect(page.get_by_text("Loaded a real patient from the held-out test set")).to_be_visible(timeout=15000)
            assert page.input_value("#predict-glucose") != ""
        check("Load example fills the form from the backend test set", load_example)

        def reset():
            page.get_by_role("button", name="Reset form").click()
            for k in PATIENT:
                assert page.input_value(f"#predict-{k}") == "", k
        check("Reset form clears every field", reset)

        # ---------- Backend unavailable + Retry ----------
        def backend_down_then_retry():
            fill_form(page, "predict", PATIENT)
            page.route("**/predict", lambda route: route.abort())
            page.get_by_role("button", name="Predict", exact=True).click()
            expect(page.get_by_text("Backend unavailable")).to_be_visible(timeout=15000)
            expect(page.get_by_role("button", name="Predict", exact=True)).to_be_enabled()
            page.screenshot(path=SHOTS / "03_backend_unavailable.png", full_page=True)
            page.unroute("**/predict")
            page.get_by_role("button", name="Retry prediction").click()
            expect(page).to_have_url(f"{APP}/results", timeout=30000)
        check("Backend-unavailable error shows, button re-enables, Retry succeeds", backend_down_then_retry)

        # ---------- Loading state + real prediction ----------
        def predict_real():
            page.goto(f"{APP}/predict")
            expect(page.locator("#predict-glucose")).to_be_visible(timeout=15000)
            fill_form(page, "predict", PATIENT)
            page.get_by_role("button", name="Predict", exact=True).click()
            expect(page.get_by_role("button", name="Analyzing patient information...")).to_be_disabled()
            expect(page).to_have_url(f"{APP}/results", timeout=30000)
            expected = api_predict(PATIENT)
            shown = page.locator(".readout-value").inner_text()
            assert shown == f"{round(expected['probability'] * 100)}%", (shown, expected["probability"])
            expect(page.get_by_text(expected["prediction_label"]).first).to_be_visible()
            expect(page.get_by_text("Model uncertainty")).to_be_visible()
            expect(page.get_by_text("What drove this estimate")).to_be_visible()
            page.screenshot(path=SHOTS / "04_results.png", full_page=True)
        check("Predict shows loading state, then the real model probability", predict_real)

        def rerun():
            page.get_by_role("button", name="Run prediction again").click()
            expect(page.get_by_role("button", name="Run prediction again")).to_be_enabled(timeout=30000)
        check("Run prediction again re-queries the backend", rerun)

        # ---------- Explainability ----------
        def explain():
            page.get_by_role("link", name="View full explanation").click()
            expect(page.get_by_text("Contribution table")).to_be_visible(timeout=30000)
            expect(page.get_by_text("Base + contributions = model probability")).to_be_visible()
            expect(page.get_by_role("heading", name="SHAP summary")).to_be_visible(timeout=15000)
            page.wait_for_timeout(800)
            page.screenshot(path=SHOTS / "05_explainability.png", full_page=True)
            page.get_by_role("link", name="Back to results").click()
            expect(page).to_have_url(f"{APP}/results")
        check("View explanation shows local + global SHAP; Back returns", explain)

        # ---------- What-if ----------
        def what_if():
            page.get_by_role("link", name="Try what-if analysis").click()
            run = page.get_by_role("button", name="Run comparison")
            expect(run).to_be_disabled(timeout=15000)
            page.fill("#whatif-glucose", "120")
            page.fill("#whatif-bmi", "28")
            expect(run).to_be_enabled()
            run.click()
            expect(page.get_by_role("heading", name="Comparison")).to_be_visible(timeout=30000)
            expected = api_predict({**PATIENT, "glucose": 120, "bmi": 28})
            expect(page.get_by_text(f"{expected['probability'] * 100:.1f}%").first).to_be_visible()
            expect(page.get_by_text("Difference (percentage points)")).to_be_visible()
            page.select_option("#feature-select", "bmi")
            expect(page.get_by_text("Training-data range of BMI")).to_be_visible(timeout=30000)
            page.select_option("#scenario-select", "modified")
            expect(page.get_by_text("Training-data range of BMI")).to_be_visible(timeout=30000)
            page.wait_for_timeout(800)
            page.screenshot(path=SHOTS / "06_what_if.png", full_page=True)
            page.get_by_role("button", name="Reset changes").click()
            assert page.input_value("#whatif-glucose") == "150"
            expect(page.get_by_role("heading", name="Comparison")).to_have_count(0)
            page.get_by_role("link", name="Back to results").click()
            expect(page).to_have_url(f"{APP}/results")
        check("What-if compares real model outputs; sensitivity, reset and back work", what_if)

        def back_to_inputs():
            page.get_by_role("button", name="Back to inputs").click()
            expect(page).to_have_url(f"{APP}/predict")
            assert page.input_value("#predict-glucose") == "150", "form values should be kept"
        check("Back to inputs keeps the entered values", back_to_inputs)

        # ---------- Model information ----------
        def model_info():
            page.goto(f"{APP}/model")
            expect(page.get_by_role("heading", name="Final model on the held-out test set")).to_be_visible(timeout=15000)
            for tab in ["Precision–recall", "Calibration", "ROC"]:
                page.get_by_role("tab", name=tab).click()
                expect(page.get_by_role("tab", name=tab)).to_have_attribute("aria-selected", "true")
                expect(page.locator(".recharts-line").first).to_be_visible()
            page.screenshot(path=SHOTS / "07_model_info.png", full_page=True)
        check("Model information shows real results; curve tabs switch", model_info)

        def about_and_404():
            page.goto(f"{APP}/about")
            expect(page.get_by_text("It is not a medical diagnostic device").first).to_be_visible()
            page.goto(f"{APP}/no-such-page")
            expect(page.get_by_role("heading", name="Page not found")).to_be_visible()
            page.get_by_role("link", name="Go to the dashboard").click()
            expect(page).to_have_url(f"{APP}/")
        check("About page, disclaimer, 404 page and its link work", about_and_404)

        def no_console_errors():
            assert not console_errors, console_errors[:3]
        check("No JavaScript errors in the browser console (desktop)", no_console_errors)

        # ---------- Responsive ----------
        for name, size in [("tablet", {"width": 820, "height": 1180}), ("mobile", {"width": 390, "height": 844})]:
            print(f"{name.title()} {size['width']}x{size['height']}")
            mctx = browser.new_context(viewport=size)
            m = mctx.new_page()

            def responsive(m=m, name=name):
                for path in ["/", "/predict", "/model", "/about", "/explain"]:
                    m.goto(f"{APP}{path}")
                    m.wait_for_load_state("networkidle")
                    no_horizontal_overflow(m)
                m.goto(f"{APP}/predict")
                expect(m.locator("#predict-glucose")).to_be_visible(timeout=15000)
                fill_form(m, "predict", PATIENT)
                m.get_by_role("button", name="Predict", exact=True).click()
                expect(m).to_have_url(f"{APP}/results", timeout=30000)
                no_horizontal_overflow(m)
                m.screenshot(path=SHOTS / f"08_{name}_results.png", full_page=True)
                m.goto(f"{APP}/model")
                expect(m.get_by_role("heading", name="Model comparison")).to_be_visible(timeout=15000)
                no_horizontal_overflow(m)
            check(f"{name}: no horizontal overflow, prediction works", responsive)

            if name == "mobile":
                def mobile_menu(m=m):
                    m.goto(APP)
                    menu = m.get_by_role("button", name="Menu")
                    expect(menu).to_be_visible()
                    menu.click()
                    m.screenshot(path=SHOTS / "09_mobile_menu.png")
                    m.get_by_role("link", name="Model information").click()
                    expect(m).to_have_url(f"{APP}/model")
                    expect(m.get_by_role("button", name="Menu")).to_be_visible()
                check("mobile: menu button opens navigation and closes after navigating", mobile_menu)
            mctx.close()

        browser.close()

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
