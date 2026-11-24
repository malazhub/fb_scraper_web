ACCESS_CODE = "12345"  # الكود الذي يجب على المستخدم إدخاله

from flask import Flask, render_template_string, request
from playwright.sync_api import sync_playwright, Page

import base64 
#import pytesseract
#import requests
from io import BytesIO
import re
import threading
#import webbrowser

import time
import cv2
import numpy as np
#import pyautogui


#from PIL import ImageChops
# استبدال النموذج السابق بنموذج قوي للمطابقة المعنوية
_transformer_model = None
_similarity_util = None
_ocr_reader = None

def get_transformer():
    global _transformer_model, _similarity_util
    if _transformer_model is None:
        from sentence_transformers import SentenceTransformer, util
        _transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
        _similarity_util = util
    return _transformer_model, _similarity_util

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en', 'ar'], gpu=False)
    return _ocr_reader

from PIL import Image
from datetime import datetime
import os

import imagehash

from playwright.sync_api import sync_playwright

import sqlite3
import json





#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def take_screenshot(page):
    page.wait_for_timeout(500)   # نصف ثانية انتظار
    return page.screenshot(full_page=False)



   
def extract_text_with_boxes(image):
    if image is None:
        print("🚫 فشل تحميل الصورة (image is None)")
        return []

    try:
        # تحويل الصورة إلى RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # استخراج بيانات النص مع مواقع الصناديق
        results = get_ocr_reader().readtext(image_rgb)
        all_texts = [res[1] for res in results]
        grouped_blocks = [[res[1]] for res in results]


        all_texts = []
        grouped_blocks = []  # ← قائمة مضافة: لتجميع البلوكات النصية

        current_block_num = -1
        current_block_text = []

        # المرور على كل كلمة في البيانات
        results = get_ocr_reader().readtext(image_rgb)
        all_texts = [res[1] for res in results]
        grouped_blocks = [[res[1]] for res in results]
        print(f"📦 عدد الكتل النصية المكتشفة: {len(all_texts)}")


        print(f"📦 عدد الكتل النصية المكتشفة: {len(all_texts)}")

        # يمكنك هنا إعادة الاثنين معًا، أو الاحتفاظ بـ grouped_blocks للاستخدام الخارجي
        # return all_texts, grouped_blocks ← في حال أردت الإثنين معًا
        return all_texts, grouped_blocks
  # ← إذا أردت فقط all_texts كما هو ظاهر

    except Exception as e:
        print("❗ خطأ أثناء تحليل الصورة:", e)
        return []



    



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







def screenshots_are_similar(img_bytes1, img_bytes2, threshold=0.98):
    # تحويل البايتات إلى صور PIL
    img1 = Image.open(BytesIO(img_bytes1)).convert("RGB")
    img2 = Image.open(BytesIO(img_bytes2)).convert("RGB")

    # حساب الـ perceptual hash لكلا الصورتين
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    # حساب الفرق بين الهاشات
    diff = hash1 - hash2
    hash_size = len(hash1.hash)**2  # عدد البتات في الهاش (عادة 64)
    
    similarity = 1 - (diff / hash_size)

    #print("🔍 pHash similarity:", similarity)
    #input("اضغط للمتابعة...")

    return similarity >= threshold



def apply_mask_to_image(image_bytes, mask):
    # تحويل الصورة من bytes إلى NumPy array بصيغة BGR (لـ OpenCV)
    image_stream = BytesIO(image_bytes)
    pil_image = Image.open(image_stream).convert("RGB")
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # التأكد أن أبعاد الصورة متوافقة مع القناع
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # إنشاء خلفية بيضاء
    background = np.ones_like(image, dtype=np.uint8) * 255

    # تحويل القناع إلى ثلاث قنوات
    if len(mask.shape) == 2:
        mask_3ch = cv2.merge([mask, mask, mask])
    else:
        mask_3ch = mask

    # تطبيق القناع
    foreground = cv2.bitwise_and(image, mask_3ch)
    background_masked = cv2.bitwise_and(background, cv2.bitwise_not(mask_3ch))
    final = cv2.add(foreground, background_masked)

    return final



def remove_background_with_rembg(image_bytes):
    raise RuntimeError('Background removal skipped on low-memory Render instance')
    # فتح الصورة من Bytes وتحويلها إلى RGB
    input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # إزالة الخلفية
    output_image = remove(input_image)

    # تحويل النتيجة إلى مصفوفة NumPy
    output_np = np.array(output_image)

    # استخراج قناة Alpha كقناع أبيض وأسود
    if output_np.shape[2] == 4:
        mask = output_np[:, :, 3]
    else:
        mask = np.ones(output_np.shape[:2], dtype=np.uint8) * 255

    return mask


def is_keyword_match(keyword, text):
    if keyword in text:
        return 1.0  # تطابق تام
    similarity = compute_similarity(keyword, text)
    return similarity


def compute_similarity(text1, text2, threshold=0.6):
    # تقسيم text2 إلى جمل أو مقاطع فرعية
    
    chunks = re.split(r'[.؟!\n]', text2)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # تضمين النصوص (text1 + كل أجزاء text2)
    texts = [text1] + chunks
    transformer_model, similarity_util = get_transformer()
    embeddings = transformer_model.encode(texts, convert_to_tensor=True)

    # حساب التشابه بين text1 وكل جزء من text2
    similarities = similarity_util.pytorch_cos_sim(embeddings[0], embeddings[1:])
    max_similarity = similarities.max().item()

    #print("...max similarity with parts of text2......", max_similarity)
    return max_similarity

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
    


def preprocess_image_for_ocr(image):
    # 1. تحويل الصورة إلى رمادية
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. زيادة التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 3. تطبيق Threshold ذكي (Adaptive Threshold) لتحويل النص إلى أسود والخلفية إلى أبيض
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY,21 , 10
    )

    # 4. ع
    processed = thresh

    return processed


import asyncio


def expand_all_facebook_posts(page):
    try:
        # 1. التقاط سكرين شوت من الصفحة (Full page screenshot غير مدعوم على الهاتف، نأخذ الصورة المعروضة فقط)
        #input("i will screen shot")
        screenshot_bytes = take_screenshot(page)


        image_stream = BytesIO(screenshot_bytes)
        pil_image = Image.open(image_stream).convert("RGB")

        # 2. تحويل الصورة إلى numpy array لـ OpenCV
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # ثم طبق زيادة التباين (مثلاً دالة preprocess_image_for_ocr التي ذكرتها لك)
        image = preprocess_image_for_ocr(image)

        # 3. استخدم pytesseract لاستخراج بيانات النصوص مع مواقع الكلمات
        #data = pytesseract.image_to_data(image, lang='eng+ara', output_type=pytesseract.Output.DICT)
        
        # 3. استخدم EasyOCR لاستخراج النصوص ومواقعها
        results = get_ocr_reader().readtext(image)  # reader هو EasyOCR reader
        targets = []

        for bbox, text, conf in results:
            text_lower = text.lower().strip()
            if text_lower in ["see more", "عرض المزيد"]:
                (x0, y0), (x1, y1), (x2, y2), (x3, y3) = bbox
                center_x = int((x0 + x1 + x2 + x3) / 4)
                center_y = int((y0 + y1 + y2 + y3) / 4)
                targets.append((center_x, center_y))
                print(f"✅ Found target '{text}' at: ({center_x}, {center_y})")



        print("targets........")
        # 5. إذا لم تجد أهداف، ارجع بعد تأخير
        if not targets:
            time.sleep(2)
            return False

        # 6. إذا وجدت، انقر على كل هدف مع فاصل 2 ثانية
        # قبل الحلقة التي تنقر فيها أضف هذه السطور لتحصل على أبعاد الصورة ونافذة المتصفح:
        img_width, img_height = pil_image.size
        viewport = page.viewport_size
        vp_width, vp_height = viewport['width'], viewport['height']

        

        for (x_img, y_img) in targets:
                device_scale_factor = page.evaluate("window.devicePixelRatio")

                x_page = int(x_img * vp_width / img_width / device_scale_factor) + 3
                y_page = int(y_img * vp_height / img_height / device_scale_factor) + 3

                print(f"Clicking at: ({x_page}, {y_page})")
                try:
                    print(f"👆 محاولة الضغط باستخدام الماوس في ({x_page}, {y_page})")
                    page.mouse.move(x_page, y_page)
                    time.sleep(0.1)
                    page.mouse.click(x_page, y_page)
                    time.sleep(2)
                except Exception as e:
                    print(f"⚠️ فشل النقر بالماوس — تجربة JavaScript: {e}")
                    page.evaluate(f"""
                        () => {{
                            const el = document.elementFromPoint({x_page}, {y_page});
                            if (el) el.click();
                        }}
                    """)

                print("Clicked")



        # 7. فاصل 2 ثانية قبل الرجوع
        time.sleep(2)
        return True

    except Exception as e:
        print(f"❌ خطأ في expand_all_facebook_posts: {e}")
        return False



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

    seen_texts = set()
    scroll_count = 0
    analyzed_count = 0
    posts_data = []
    last_screenshot = None
    last_dom_snapshot = ""
    
   
   
    smart_scroll_mobile(page)
    smart_scroll_mobile(page)
    smart_scroll_mobile(page)
    smart_scroll_mobile(page)

    
    while analyzed_count < max_posts and scroll_count < max_scrolls:
        
        
        
        current_dom_snapshot = get_dom_snapshot(page)

        #if current_dom_snapshot != last_dom_snapshot:
        #    print("📄 تم تحميل محتوى جديد")
        expanded = expand_all_facebook_posts(page)
            #smart_scroll_mobile(page)
            #page.wait_for_timeout(1500)
            
        #    last_dom_snapshot = current_dom_snapshot
       # else:
        #    print("🟡 نفس المحتوى السابق — لن ننفذ expand")



        # ثم نأخذ screenshot كالمعتاد
        screenshot_bytes = page.screenshot(full_page=False)





      
        if last_screenshot and screenshots_are_similar(screenshot_bytes, last_screenshot):
            print("[Scroll Detection] Screenshot is similar. Skipping...")
            scroll_count += 1
            if 2 <= scroll_count <= 4:

                # حفظ الموقع الحالي
                

                # الصعود لأعلى الصفحة
                
                scroll_up_and_return_smoothly(page)
                #scroll_up_and_return_smoothly(page)
                            
            continue
        else:
            #print("[Scroll Detection] First screenshot.")
            #input("111111111")
            
            scroll_count = 0
            last_screenshot = screenshot_bytes
            print(scroll_count)
        masked_image = None  # تأكد من وجود المتغير قبل try

        try:
            mask = remove_background_with_rembg(screenshot_bytes)
            masked_image = apply_mask_to_image(screenshot_bytes, mask)
        except Exception as e:
            print("Mask extraction failed:", e)
            try:
                image_stream = BytesIO(screenshot_bytes)
                pil_image = Image.open(image_stream).convert("RGB")
                masked_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e2:
                print("Fallback failed:", e2)
                masked_image = None

        
        if masked_image is not None:
            _, buffer = cv2.imencode('.png', masked_image)
            image_base64 = base64.b64encode(buffer).decode("utf-8")
        else:
            image_base64 = ""


        all_texts, grouped_blocks = extract_text_with_boxes(masked_image)

        #print("...all text...",all_texts)
        #input(".....")

        for i, full_text in enumerate(all_texts):

            #print("...full text...",full_text)
            #input("......")
            for keyword in keywords:
                similarity = compute_similarity(full_text.strip(), keyword.strip())
                #print("...similarity....",similarity)
                #input("......")
                if similarity >= similarity_threshold:

                    
                    
                    matched_block = grouped_blocks[i]
                    #print("✅ Bloc مطابق:", ' '.join(matched_block))
                    #input(".....")

                   # ترميز الصورة إلى base64
                    image_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

                    posts_data.append({
                        "text": ' '.join(matched_block),
                        "matched_keyword": keyword,
                        "similarity": f"{similarity:.2f}",
                        "image_base64": image_base64,
                        "page_url": page.url
                    })

                    break


        analyzed_count += 1
        if analyzed_count < max_posts:
            smart_scroll_mobile(page)
            #page.wait_for_timeout(1500)





    posts_data.sort(key=lambda x: x.get("similarity", 0), reverse=True)


    return posts_data





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





def home():

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT dynamically
    app.run(host="0.0.0.0", port=port)


