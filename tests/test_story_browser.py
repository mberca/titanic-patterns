"""Optional visual regressions; requires Playwright and a local browser.

Run with a Playwright-enabled Python: python -m unittest discover -s tests
-p test_story_browser.py. Windows uses installed Edge; other platforms use Chromium.
"""

import sys
import unittest
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sync_playwright, "Optional Playwright dependency is not installed")
class VoyageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            channel="msedge" if sys.platform == "win32" else None
        )

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def tearDown(self):
        for context in self.browser.contexts:
            context.close()

    def open_story(self, width, motion="no-preference"):
        page = self.browser.new_page(
            viewport={"width": width, "height": 1000},
            reduced_motion=motion,
            color_scheme="light",
        )
        page.goto((ROOT / "docs" / "index.html").as_uri() + "?scoutTheme=light")
        return page

    def test_liner_glides_once_and_wake_finishes(self):
        page = self.open_story(1600)
        liner = page.locator(".voyage-liner")
        first_x = liner.bounding_box()["x"]
        page.wait_for_timeout(4200)
        self.assertGreater(liner.bounding_box()["x"], first_x + 10)
        self.assertTrue(
            page.locator(".hero-voyage").evaluate(
                """el => {
                    const animations = el.getAnimations({subtree:true});
                    return animations.length === 3 && animations.every(a =>
                        a.playState === "finished" &&
                        a.effect.getComputedTiming().iterations === 1 &&
                        a.effect.getComputedTiming().endTime <= 4000);
                }"""
            )
        )
        self.assertTrue(
            page.locator(".voyage-wake").evaluate_all(
                "els => els.every(el => parseFloat(getComputedStyle(el).strokeDashoffset) === 0)"
            )
        )

    def test_voyage_stays_clear_of_text_at_mobile_tablet_and_desktop_widths(self):
        for width in (320, 390, 768, 1050, 1280, 1600):
            with self.subTest(width=width):
                page = self.open_story(width)
                art = page.locator(".hero-voyage").bounding_box()
                for selector in ("h1", ".eyebrow", ".lead", ".hero > .aside"):
                    text = page.locator(selector).first.bounding_box()
                    overlaps = (
                        art["x"] < text["x"] + text["width"]
                        and art["x"] + art["width"] > text["x"]
                        and art["y"] < text["y"] + text["height"]
                        and art["y"] + art["height"] > text["y"]
                    )
                    self.assertFalse(overlaps, f"Voyage overlaps {selector}")
                self.assertFalse(
                    page.evaluate("document.documentElement.scrollWidth > innerWidth")
                )
                page.close()

    def test_reduced_motion_retains_static_art_and_theme_controls_still_work(self):
        page = self.open_story(390, "reduce")
        art = page.locator(".hero-voyage")
        self.assertEqual(art.get_attribute("aria-hidden"), "true")
        self.assertEqual(art.get_attribute("focusable"), "false")
        self.assertEqual(
            art.evaluate("el => el.getAnimations({subtree:true}).length"), 0
        )
        self.assertGreater(
            page.locator(".voyage-liner").evaluate(
                "el => Number(getComputedStyle(el).opacity)"
            ),
            0,
        )
        hull = page.locator(".liner-hull")
        original = hull.evaluate("el => getComputedStyle(el).fill")
        page.click("#theme")
        self.assertNotEqual(hull.evaluate("el => getComputedStyle(el).fill"), original)
        page.select_option("#language", "Python")
        self.assertIn("Python", page.locator("#model-language").inner_text())

    def test_people_analytics_context_is_readable_before_the_first_chart(self):
        for width in (320, 390, 1600):
            with self.subTest(width=width):
                page = self.open_story(width, "reduce")
                bridge = page.locator("#people-analytics-bridge")
                bridge.scroll_into_view_if_needed()
                self.assertTrue(bridge.is_visible())
                self.assertEqual(
                    page.locator("#people-analytics-title").evaluate(
                        "el => el.tagName"
                    ),
                    "H2",
                )
                self.assertIn(
                    "not a model of employee behaviour or guidance for workplace decisions",
                    bridge.inner_text(),
                )
                bounds = bridge.bounding_box()
                chart = page.locator("#rates").bounding_box()
                self.assertLess(bounds["y"] + bounds["height"], chart["y"])
                self.assertGreaterEqual(bounds["x"], 0)
                self.assertLessEqual(bounds["x"] + bounds["width"], width)
                self.assertFalse(
                    page.evaluate("document.documentElement.scrollWidth > innerWidth")
                )
                page.close()
