import asyncio
from playwright.async_api import async_playwright

async def scrape_page(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--ignore-certificate-errors"])
            context = await browser.new_context()
            page = await context.new_page()

            await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            if await page.query_selector("iframe[title*='challenge']"):
                print(f"CAPTCHA detected on {url}. Skipping.")
                return "CAPTCHA detected, skipping."
            
            content = await page.evaluate("document.body.innerText")
            
            await browser.close()
            
            return content.strip()
    except Exception as e:
        return f"Error scraping {url}: {e}"

async def scrape_multiple(urls):
    tasks = [scrape_page(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return "\n\n".join(results)
