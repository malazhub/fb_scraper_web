ACCESS_CODE = "12345"  # الكود الذي يجب على المستخدم إدخاله

from flask import Flask, render_template_string, request
from playwright.sync_api import sync_playwright, Page

import base64 
#import pytesseract
#import requests
import re
import threading
#import webbrowser

import time
#import pyautogui


#from PIL import ImageChops
# استبدال النموذج السابق بنموذج قوي للمطابقة المعنوية
from datetime import datetime
import os


from playwright.sync_api import sync_playwright

import sqlite3
import json





#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def take_screenshot(page):
    page.wait_for_timeout(500)   # نصف ثانية انتظار
    return page.screenshot(full_page=False)



   
def extract_and_match_posts(all_texts, keywords, similarity_threshold):
    results = []

    # ✅ تحقق أولاً أن all_texts صالحة
    if not all_texts:
        print("⚠️ لا توجد نصوص لتحليلها (all_texts is empty or None).")
        return []

    

    # تأكد أن كل عنصر نصي
    try:
        texts = [t.strip() for t in all_texts if isinstance(t, str)]
    except Exception as e:
        print(f"❗ خطأ أثناء تجهيز النصوص: {e}")
        return []

    ##print("....texts....", texts)
    ##input(".......")

    for line in texts:
        if line.strip() == "":
            continue

        #
        #print(f"\n--- Individual Line ---\n{line}\n")

        for kw in keywords:
            sim = compute_similarity(kw, line)

            if sim >= similarity_threshold:
                results.append({
                    "matched_keyword": kw,
                    "similarity": sim,
                    "text": line
                })
                break  # ✅ توقف بعد أول تطابق ناجح

    return results




def split_into_blocks (text):
    return [block.strip() for block in re.split(r'\n\s*\n', text) if block.strip()]

def is_similar_ai(block, keywords, threshold=0.6):
    best_similarity = 0.0
    best_keyword = None
    for keyword in keywords:
        similarity = compute_similarity(block, keyword)
        if similarity > best_similarity:
            best_similarity = similarity
            best_keyword = keyword
    if best_similarity >= threshold:
        return True, best_similarity, best_keyword
    return False, 0.0, None

def extract_matching_blocks(text, keywords):
    blocks = split_into_blocks(text)
    matched = []
    for block in blocks:
        match, sim, keyword = is_similar_ai(block, keywords)
        if match:
            matched.append((block, sim, keyword))
    return matched


# إعداد Flask
app = Flask(__name__)







def is_keyword_match(keyword, text):
    if keyword in text:
        return 1.0  # تطابق تام
    similarity = compute_similarity(keyword, text)
    return similarity


def compute_similarity(text1, text2, threshold=0.6):
    """Lightweight lexical similarity without loading a machine-learning runtime."""
    import difflib
    import unicodedata

    def tokens(value):
        normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
        normalized = "".join(
            char for char in normalized
            if unicodedata.category(char) != "Mn"
        )
        return re.findall(r"\w+", normalized)

    query = tokens(text1)
    words = tokens(text2)
    if not query or not words:
        return 0.0
    available = set(words)
    scores = []
    for word in set(query):
        scores.append(1.0 if word in available else max(
            (difflib.SequenceMatcher(None, word, other).ratio() for other in available),
            default=0.0
        ))
    return sum(scores) / len(scores)


def save_cookies(context, email):
    cookies = context.cookies()
    if not cookies:  # إذا كانت القائمة فارغة، لا تحفظ
        print(f"⚠️ لا توجد كوكيز للحفظ للمستخدم: {email}")
        return
    cookie_json = json.dumps(cookies)
    try:
        conn = sqlite3.connect("/tmp/sessions.db")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS cookies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                cookie_json TEXT
            )
        """)
        conn.execute("REPLACE INTO cookies (user_id, cookie_json) VALUES (?, ?)", (email, cookie_json))
        conn.commit()
    except Exception as e:
        print(f"❌ خطأ أثناء حفظ الكوكيز للمستخدم {email}: {e}")
    finally:
        conn.close()


def load_cookies(context, email):
    try:
        conn = sqlite3.connect("/tmp/sessions.db")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS cookies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                cookie_json TEXT
            )
        """)
        cursor = conn.execute("SELECT cookie_json FROM cookies WHERE user_id = ?", (email,))
        row = cursor.fetchone()
        if row:
            cookies = json.loads(row[0])
            context.add_cookies(cookies)
        else:
            print(f"⚠️ لا توجد كوكيز محفوظة للمستخدم: {email}")
    except Exception as e:
        print(f"❌ خطأ أثناء تحميل الكوكيز للمستخدم {email}: {e}")
    finally:
        conn.close()


def get_dom_snapshot(page):
    try:
        return page.content()
    except Exception as e:
        print("❌ فشل قراءة DOM:", e)
        return ""
    


def expand_all_facebook_posts(page):
    """Expand visible Facebook posts using DOM text controls instead of OCR."""
    changed = False
    for label in ("See more", "عرض المزيد"):
        try:
            controls = page.get_by_text(label, exact=True)
            for index in range(min(controls.count(), 12)):
                try:
                    controls.nth(index).click(timeout=1200)
                    changed = True
                    page.wait_for_timeout(150)
                except Exception:
                    continue
        except Exception as exc:
            print(f"Could not expand '{label}': {exc}")
    return changed


def smart_scroll_mobile(page):
    #page.evaluate("window.scrollBy(0, window.innerHeight * 2);")
    #page.wait_for_timeout(3000)
    page.evaluate("""
        const el = document.querySelector('[role="feed"]') || document.querySelector('div[data-pagelet]');
        if (el) {
            el.scrollTop -= 200;  // Scroll up قليلاً
            el.scrollTop += 200;  // Scroll down قليلاً

        } else {
            window.scrollBy(0, -200);
            window.scrollBy(0, 200);
        }
    """);
    page.wait_for_timeout(300);

    page.evaluate("window.scrollBy(0, window.innerHeight * 1);")
    page.wait_for_timeout(500);
    return 


import random
import time

def scroll_up_and_return_smoothly(page):
    import random

    # 1. حفظ الموقع الحالي
    current_scroll = page.evaluate("() => window.scrollY")
    print(f"📌 الموقع الحالي المحفوظ: {current_scroll}")
    if current_scroll <= 0:
       print("⚠️ الموقع المحفوظ هو أعلى الصفحة. سيتم تخطي العودة.")
       return


    # 2. الصعود لأعلى الصفحة بـ 20 خطوة تدريجية
    scroll_steps = random.randint(10, 20)
    for i in range(scroll_steps):
        page.evaluate("window.scrollBy(0, -window.innerHeight / 2);")
        wait = random.randint(100, 300)
        page.wait_for_timeout(wait)

        if i % 5 == 0:
            pause = random.randint(500, 1500)
            print(f"⏸️ توقف قصير: {pause}ms")
            page.wait_for_timeout(pause)

    print("⬆️ تم الصعود لأعلى الصفحة.")
    scroll_fraction = random.uniform(0.5, 1.2)  # من نصف إلى 1.2 من ارتفاع الشاشة
    page.evaluate(f"window.scrollBy(0, window.innerHeight * {scroll_fraction});")
    page.wait_for_timeout(random.randint(300, 700))

    page.evaluate(f"window.scrollBy(0, -window.innerHeight * {scroll_fraction});")

    
    
    # توقف إضافي يشبه التأمل أو التفكير
    if random.random() < 0.3:
        pause = random.randint(800, 1500)
        print(f"🤔 توقف إضافي للتأمل لمدة {pause}ms...")
        page.wait_for_timeout(pause)


    # 3. تصرفات بشرية خفيفة بدل النقر
    for i in range(3):
        x = random.randint(50, 300)
        y = random.randint(100, 600)

        # تحريك الماوس بشكل خفيف
        for step in range(5):
            move_x = x + random.randint(-5, 5)
            move_y = y + random.randint(-5, 5)
            page.mouse.move(move_x, move_y)
            page.wait_for_timeout(random.randint(30, 60))

        # محاكاة قراءة أو تمعن
        page.wait_for_timeout(random.randint(400, 800))

        # تمرير بسيط عشوائي جداً (شبيه بلمس إصبع)
        delta_y = random.randint(-50, 50)
        page.evaluate(f"window.scrollBy(0, {delta_y});")
        print(f"🤏 تمرير بشري خفيف بمقدار {delta_y}px")
        page.wait_for_timeout(random.randint(300, 700))
        page.evaluate(f"window.scrollBy(0, {-delta_y});")

    # 4. تمرير ذكي للأسفل
    smart_scroll_mobile(page)
    page.evaluate("window.scrollBy(0, -window.innerHeight);")


    # 5. العودة تدريجيًا إلى الموقع الأصلي
    steps = random.randint(10, 20)
    step_size = current_scroll // steps

    for i in range(steps):
        target_scroll = (i + 1) * step_size
        page.evaluate(f"window.scrollTo(0, {target_scroll});")
        wait = random.choice([90, 120, 150, 180])

        page.wait_for_timeout(wait)

    print("⬇️ تم الرجوع تدريجيًا إلى الموقع الأصلي.")




def analyze_posts_via_screenshot_ai(page: Page, keywords, max_posts, max_scrolls, similarity_threshold):
    """Analyze post text in the DOM and capture screenshots for matched posts."""
    results = []
    seen = set()
    max_posts = max(1, int(max_posts))

    for scroll_index in range(max(1, int(max_scrolls))):
        expand_all_facebook_posts(page)
        cards = page.locator('[role="article"]')
        candidates = []
        try:
            count = cards.count()
        except Exception:
            count = 0

        if count:
            for index in range(count):
                card = cards.nth(index)
                try:
                    text = card.inner_text(timeout=3000).strip()
                    if len(text) >= 30:
                        candidates.append((card, text))
                except Exception:
                    continue
        else:
            try:
                body = page.locator("body").inner_text(timeout=5000)
                candidates.extend((None, block) for block in split_into_blocks(body) if len(block) >= 50)
            except Exception as exc:
                print(f"Could not read Facebook page text: {exc}")

        found_new = False
        for card, post_text in candidates:
            key = " ".join(post_text.casefold().split())
            if not key or key in seen:
                continue
            seen.add(key)
            found_new = True

            best_keyword, best_score = None, 0.0
            for keyword in keywords:
                keyword = keyword.strip()
                if keyword:
                    score = compute_similarity(keyword, post_text)
                    if score > best_score:
                        best_keyword, best_score = keyword, score

            if best_keyword is None or best_score < similarity_threshold:
                continue

            try:
                shot = card.screenshot(timeout=5000) if card is not None else page.screenshot(full_page=False)
            except Exception:
                shot = page.screenshot(full_page=False)
            results.append({
                "text": post_text,
                "matched_keyword": best_keyword,
                "similarity": f"{best_score:.2f}",
                "image_base64": base64.b64encode(shot).decode("utf-8"),
                "page_url": page.url
            })
            if len(results) >= max_posts:
                return sorted(results, key=lambda item: float(item["similarity"]), reverse=True)

        if scroll_index > 0 and not found_new:
            break
        try:
            page.evaluate("window.scrollBy(0, Math.max(500, window.innerHeight * 0.85))")
            page.wait_for_timeout(1100)
        except Exception:
            break

    return sorted(results, key=lambda item: float(item["similarity"]), reverse=True)


def run_facebook_scraper(keywords, facebook_pages, max_posts, similarity_threshold, email):
    max_scrolls = 5

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--single-process",
                "--disable-gpu"
            ]
        )




        context = browser.new_context(
            user_agent="Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36"
        )

        load_cookies(context, email)

        page = context.new_page()
        all_results = []

        for fb_page in facebook_pages:
            print(f"\n🔗 فتح الرابط: {fb_page}")
            fb_page = fb_page.replace("www.facebook.com", "m.facebook.com")

            try:
                page.goto(fb_page, timeout=120000)
                print("✅ تم تحميل الصفحة بنجاح")
            except Exception as e:
                print("❌ فشل في تحميل الصفحة:", e)
                continue

            print("🔒 التحقق من تسجيل الدخول...")
            try:
                search_input = page.locator("input[name='query']")
                print("⏳ محاولة العثور على شريط البحث...")
                search_input.wait_for(state="visible", timeout=5000)
                print("✅ تم العثور على شريط البحث")
                search_input.fill(keywords[0])
                search_input.press("Enter")
                page.wait_for_timeout(5000)
            except Exception as e:
                print("⚠️ لم يتم العثور على شريط البحث — نتابع التحليل مباشرة (قد تكون مجموعة):")
                print(f"🔎 الخطأ: {e}")

            smart_scroll_mobile(page)
            smart_scroll_mobile(page)
            page.keyboard.press("PageDown")
            page.keyboard.press("PageDown")



            print("✨ محاولة توسيع المنشورات...")
            try:
                expanded = expand_all_facebook_posts(page)
                if not expanded:
                    print("⚠️ لم يتم العثور على زر 'عرض المزيد' — نتابع التحليل.")
            except Exception as e:
                print("⚠️ فشل في محاولة توسيع المنشورات:", e)


            print("🎯 متابعة تحليل المنشورات...")
            try:
                posts_data = analyze_posts_via_screenshot_ai(
                    page, keywords, max_posts, max_scrolls, similarity_threshold
                )
                print(f"📊 عدد المنشورات المستخرجة: {len(posts_data)}")
                all_results.extend(posts_data)

            except Exception as e:
                if 'posts_data' in locals() and posts_data:
                    print(f"⚠️ سيتم الاحتفاظ بـ {len(posts_data)} منشورًا تم استخراجه قبل الخطأ.")
                    all_results.extend(posts_data)
                else:
                    print("⚠️ لم يتم استخراج أي منشورات قبل حدوث الخطأ.")

            save_cookies(context, email)

        browser.close()
        return all_results

        




def cleanup_static_images():
    folder = "static"
    for filename in os.listdir(folder):
        if filename.startswith("match_") and filename.endswith(".png"):
            os.remove(os.path.join(folder, filename))


# واجهة HTML بسيطة
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>

<script>
function openImage(src) {
  const w = window.open("", "_blank", "width=900,height=600,resizable=yes");
  w.document.write(`
    <html><head><title>Enlarged Image</title></head>
    <body style="margin:0;display:flex;justify-content:center;align-items:center;background:#000;">
      <img src="${src}" style="max-width:100%;max-height:100%;">
    </body></html>
  `);
}
</script>

    <meta charset="UTF-8">
    <title>Facebook Post Analyzer</title>
</head>
<body>
    <h2>Analyze Facebook Posts   /  malaz janbeih-malazjanbeih@gmail.com-+96178719611  </h2>
    <form method="POST">
        <label>User Email or ID:</label><br>
        <input type="text" name="email" required><br><br>

        <label>Facebook Links (one per line):</label><br>
        <textarea name="links" rows="10" cols="60" style="border:1px solid #ccc; padding:5px; line-height:1.5; font-family: monospace; white-space: pre-wrap;"></textarea><br><br>
        <label>Keywords (separated by -):</label><br>
        <input type="text" name="keywords" size="60"><br><br>
        <label>Max Posts per Link:</label><br>
        <input type="number" name="max_posts" min="1" placeholder="Enter max posts"><br><br>
        <label>Similarity Threshold (default 0.8):</label><br>
        <input type="number" name="similarity" step="0.1" min="0.5" max="1.0" value="0.8"><br><br>
        <label>Enter Access Code:</label><br>
        <input type="text" name="access_code" required><br><br>

        <input type="submit" value="Analyze">
    </form>

   {% if message %}
    <p style="color:red;"><strong>{{ message }}</strong></p>
    {% endif %}

    {% for r in results %}
        <li>
            <strong>Matched Keyword:</strong> {{ r.matched_keyword }}<br>
            <strong>Text:</strong> <pre>{{ r.text }}</pre><br>
            <img src="data:image/png;base64,{{ r.image_base64 }}" width="300"
                style="cursor: zoom-in; border: 1px solid #ccc; border-radius: 5px;"
                onclick="openImage(this.src)">
        </li><hr>
    {% endfor %}

</body>
</html>
'''



@app.route("/", methods=["GET", "POST"])
def home():
    message = ""
    results = []

    if request.method == "POST":
        access_code = request.form.get("access_code", "").strip()
        if access_code != ACCESS_CODE:
            message = "❌ Incorrect Access Code."
            return render_template_string(HTML_TEMPLATE, message=message)

        links = request.form["links"].strip().splitlines()
        keywords = request.form["keywords"].strip().split("-")
        max_posts = int(request.form["max_posts"])
        similarity_input = request.form.get("similarity", "0.7")
        try:
            similarity_threshold = float(similarity_input)
        except:
            similarity_threshold = 0.7

        email = request.form.get("email", "").strip()

        results = run_facebook_scraper(keywords, links, max_posts, similarity_threshold, email)

        if not results:
            message = "❗ No matched keyword found."
        return render_template_string(HTML_TEMPLATE, results=results, message=message)

    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT dynamically
    app.run(host="0.0.0.0", port=port)


