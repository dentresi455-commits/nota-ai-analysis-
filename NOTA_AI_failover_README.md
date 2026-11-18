# NOTA AI — AI failover tuzatish to‘plami (2026-11-18)

## Holat

Ushbu to‘plamda Render `nota-ai-analysis-` loyihasining `app.py` faylidagi faqat `pdf_to_notes` funksiyasini **Gemini → OpenRouter → Groq** zaxira marshrutiga almashtiradigan xavfsiz patch skript va 7 ta offline sinov bor.

**Muhim:** Kod hozir production saytga joriy qilinmagan. GitHub integratsiyasi `403 Resource not accessible by integration` javobini qaytargani sababli repoga yozib bo‘lmadi. Ulangan GitHub integratsiyasiga aynan `dentresi455-commits/nota-ai-analysis-` repository’sida write huquqi kerak. Bu to‘plam joriy etilmasdan “real ishlayapti” deb aytilmaydi.

## O‘rnatish (Render backend repository)

1. `nota_ai_render_failover_patch.py` va `test_nota_ai_render_failover.py` fayllarini repository ildiziga (`app.py` yoniga) joylang.
2. Mavjud `app.py` va kalitlarni zaxiralang; test/preview branch ishlating.
3. `python nota_ai_render_failover_patch.py` — skript `app.py.backup-before-ai-failover` nusxasini saqlaydi, `pdf_to_notes` funksiyasini almashtiradi va sintaksisni tekshiradi.
4. `python -m unittest -v test_nota_ai_render_failover.py` — 7 offline sinovni ishga tushiradi.
5. Render xizmatining **server-side** environment variables sozlamalarida `GEMINI_API_KEY`, `OPENROUTER_API_KEY` va `GROQ_API_KEY` mavjudligini tekshiring. API kalitlarini frontendda yoki git kodida saqlamang; suhbatga ham yubormang.
6. Maxsus test muhitida haqiqiy PDF nota va mos MP3 bilan integratsion test bajaring; har bir provayderda real model, limit, fayl va javob JSON ni tekshiring.
7. Faqat yuqoridagi testlar muvaffaqiyatli bo‘lgach production’ga merge va deploy qiling. `POST /v1/analyze-files` natijasidagi `score.ai_provider` qaysi AI ishlaganini qaytaradi. Render’ning Health Check Path sozlamasini `/health` ga moslang.

## Yechilgan muammo

Oldingi Render `pdf_to_notes` funksiyasi faqat `GEMINI_API_KEY` bilan so‘rov yuborardi va Gemini 4xx/5xx yoki invalid JSON qaytarganda `502` holatiga tushardi. Patch bir urinishdan so‘ng keyingi sozlangan provayderga o‘tadi. Har bir urinish 15 soniya bilan cheklanadi, notalar/vaqt voqealari tekshiriladi va tasdiqlanmagan baho yaratilmaydi.

**Cheklov:** Groq’ning ayni foydalanilgan vision modeli bir so‘rovda maksimum 3 rasm qabul qiladi. 4 sahifali nota bo‘lsa, uning sahifalarini tushirib tashlash o‘rniga Groq bosqichi o‘tkazib yuboriladi. OpenRouter va Groq API xarajatlari bo‘lishi mumkin.

## Qo‘shimcha xatarlar va hal etilmagan vazifalar

- Vercel production kodidagi `src/lib/nota-ai/router.mjs` ning ko‘rilgan versiyasida uchinchi AI `openai`, `groq` emas; production’da **alohida tuzatish** talab etiladi. Render patchi Vercel routerini o‘zgartirmaydi.
- Vercel environment variables ichida `GROQ_API_KEY` faqat `preview` uchun ko‘ringan. Production uchun kalit to‘g‘ri va ruxsatli ekanini tekshirib alohida sozlash kerak.
- Render `nota-ai-analysis` xizmati `free` rejasida, `1` instansiyada. Render dokumentatsiyasiga ko‘ra 15 daqiqa faolsizlikdan so‘ng uyqu holatiga o‘tadi. Bu muammoni to‘liq bartaraf etish uchun doimiy ishlaydigan tarifga o‘tish (to‘lovli o‘zgarish) kerak bo‘ladi.
- 2026-11-15 Render jurnallarida `POST /v1/analyze-files` -> 502 holatlari qayd etilgan. Amaldagi AI provayderlar uchun jonli 429/502/timeout sinovlari hali bajarilmagan.
- Supabase’da faol loyihalar ko‘rindi, ammo NOTA AI ishlab turgan database aynan ulardan foydalanishini tekshirish kerak; Vercel’da `DATABASE_URL` ham mavjud.

## Kod va testlar

- `nota_ai_render_failover_patch.py` — faqat bitta funksiyani almashtiradi; fayl struktura mos bo‘lmasa hech narsa o‘zgartirmaydi.
- `test_nota_ai_render_failover.py` — haqiqiy API so‘rovlariga chiqmaydigan 7 ta `unittest`. Offline test 2026-11-18 kuni 7/7 PASS.

## Rollback

Xato bo‘lsa avvalgi versiyaga qaytish uchun zaxira fayldan `app.py` ni tiklang yoki GitHub commit rollback qiling. Production deploy’ni amalga oshirishdan oldin test orqali oldingi foydalanuvchi ish jarayonlari (PDF/audio tahlil, hisobga kirish, tarix va kredit) ham tekshirilishi kerak.
