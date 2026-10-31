import base64
import os 
from dotenv import load_dotenv
load_dotenv()

from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI(title="SmartNote AI API")

# آدرس‌های محیط توسعه فرانت‌اند
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_FILES = 8
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

NOTE_INSTRUCTIONS = """
تو یک دستیار تخصصی و بسیار دقیق جزوه‌نویسی هستی. وظیفه‌ات فقط تبدیل عکس‌های تخته کلاس به جزوه‌ای وفادار، کامل و منظم به زبان فارسی است.

قواعد ضروری:
1) همه تصاویر را با دقت بررسی کن و ترتیب آن‌ها را همان ترتیب ارسال‌شده در نظر بگیر.
2) متن، تیترها، تعریف‌ها، مثال‌ها، اعداد، واحدها، نمادها و فرمول‌ها را تا حد امکان دقیق بازنویسی کن.
3) هیچ فرمول، عدد، اسم، رابطه یا جمله ناخوانایی را حدس نزن. هر قسمت نامطمئن را دقیقاً با برچسب «[ناخوانا در تصویر؛ نیازمند بررسی]» مشخص کن.
4) اگر بخشی از تخته در عکس بعدی ادامه پیدا کرده، مطالب را به هم وصل کن و تکرارها را حذف کن؛ اما هیچ نکته تازه‌ای از خودت اضافه نکن.
5) توضیح کوتاه آموزشی فقط وقتی اضافه کن که برای فهم نوشته‌های تخته لازم باشد؛ آن را با عنوان «توضیح تکمیلی» از متن اصلی جدا کن.
6) فرمول‌ها را در خط جداگانه بنویس و نمادها و توان‌ها را حفظ کن. اگر از نمایش ریاضی مطمئن نیستی، آن را با برچسب ناخوانا مشخص کن.
7) جزوه را با عنوان مناسب، تیترهای سطح‌بندی‌شده، فهرست‌ها و فاصله‌گذاری خوانا تنظیم کن.
8) خروجی را فارسی، راست‌چین‌پسند و مناسب چاپ یا تبدیل به PDF بنویس.
9) مقدمه، تعریف درباره خودت یا توضیح روند کارت ننویس؛ مستقیماً جزوه را ارائه بده.
10) در پایان، بخشی با عنوان «مواردی که بهتر است با عکس اصلی بررسی شوند» بیاور. اگر مورد نامطمئنی وجود ندارد، بنویس «مورد مشخصی شناسایی نشد».
""".strip()


@app.get("/health")
def health():
    return {"status": "ok", "service": "SmartNote AI"}


@app.post("/api/generate-notes")
async def generate_notes(images: List[UploadFile] = File(...)):
    if not images:
        raise HTTPException(status_code=400, detail="حداقل یک تصویر انتخاب کنید.")
    if len(images) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"حداکثر {MAX_FILES} تصویر مجاز است.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="کلید API روی سرور تنظیم نشده است. متغیر OPENAI_API_KEY را تنظیم کن.",
        )

    content = [{"type": "input_text", "text": NOTE_INSTRUCTIONS}]

    for image_file in images:
        if image_file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"فرمت فایل {image_file.filename or 'بدون نام'} پشتیبانی نمی‌شود. فقط JPG، PNG و WEBP مجازند.",
            )

        raw = await image_file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="یکی از تصاویر خالی است.")
        if len(raw) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"حجم فایل {image_file.filename or 'بدون نام'} بیشتر از ۱۰ مگابایت است.",
            )

        encoded = base64.b64encode(raw).decode("utf-8")
        data_url = f"data:{image_file.content_type};base64,{encoded}"
        # برای خواندن نوشته‌های ریز تخته، جزئیات اصلی تصویر حفظ شود.
        content.append({"type": "input_image", "image_url": data_url, "detail": "high"})

    try:
        client = OpenAI(api_key=api_key, timeout=180.0, max_retries=2)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-6.1-sol"),
            reasoning={"effort": os.getenv("OPENAI_REASONING_EFFORT", "high")},
            input=[{"role": "user", "content": content}],
        )
        notes = (response.output_text or "").strip()
        if not notes:
            raise RuntimeError("مدل خروجی متنی برنگرداند.")
        return {"notes": notes}
    except Exception as exc:
        # جزئیات فنی در ترمینال سرور می‌ماند و به کاربر نمایش داده نمی‌شود.
        print(f"SmartNote generation error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="تولید جزوه انجام نشد. اتصال اینترنت، اعتبار API و نام مدل را بررسی کن.",
        )
