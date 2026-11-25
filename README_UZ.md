# NOTA AI — nota/audio/Gemini va API vaqt cheklovlari uchun sinov yangilanishi

**Sana:** 2026-11-25. **Maqsad:** mavjud NOTA AI kodini saqlagan holda 15 daqiqagacha audio bo‘laklash, >300/500 nota hodisalari yo‘qolishini oldini olish, Gemini 503 uchun nazoratli qayta urinish hamda Vercel 60/120 s cheklovini yumshatish.

## Qanday tekshirildi

- Haqiqiy `nota-ai-analysis-` repozitoriysining `omr-v3-test-20261121` tarmog‘idagi `app.py`, `advanced_analysis.py`, `score_omr_v3.py` kodiga qarab tuzildi.
- `score_omr_v3.py` asosining Git blob SHA: `d433af7bdb7bd07525c4b1b13560d1fa4065a1f9` (2026-11-25 kuni tekshirilgan).
- Avtomatik testlar: **28/28**, shu jumladan 900 sekund davomli audio bilan 500 dan ortiq nota hodisalarini tekshirish va 503→backup model **imitatsiya qilingan** testi.
- Yangi kod **GitHub’ga/Render’ga/Vercel’ga hali chiqarilmagan**; ChatGPT GitHub integratsiyasi yozishga 403 qaytaryapti. `main` va production versiyalarga tegilmagan.

## Nimalar o‘zgaradi

| Muammo | Tuzatish | Muhim cheklov |
|---|---|---|
| 500 ta audio hodisasi | `advanced_analysis.py`dagi `events[:500]` olib tashlanadi | Monofonik ovoz aniqligi cheklangan |
| 300 ta solishtirish | Butun qayd etilgan voqealar bo‘yicha xotira-tejamkor Hirschberg solishtirish va uzoq audio uchun avtomatik 30s bo‘laklar | Juda katta, murakkab ketma-ketliklar xato qaytaradi, taxminiy baho to‘qilmaydi |
| 900 s audio | `long_audio.py` 30 soniyalik bo‘laklarni qayta ishlaydi, o‘zi o‘tadi, vaqtni tiklaydi | Vercel fayl yuklash/proksi chegarasi alohida |
| Gemini 503 | Dastlabki 503/5xx uchun jitterli backoff, operator sozlasa backup Gemini modeli | Google serveri yoki kvota sababli xatolarni 100% yo‘qotib bo‘lmaydi |
| Vercel 60/120 | Marshrutlarning mavjud manba kodida `maxDuration=300`, 220/300 nota parser limiti ko‘tariladi | To‘liq 15 daqiqalik, fondagi tahlil hali tayyor emas |

## Backend’ga qo‘shish — FAQAT STAGING

GitHub: `dentresi455-commits/nota-ai-analysis-`, tarmoq: `omr-v3-test-20261121`.

1. ZIP ichidagi `backend/long_audio.py`, `backend/long_alignment.py`, `backend/score_omr_v3.py`, `backend/photo_preprocessing.py` fayllarini ushbu repozitoriyning **ildiz papkasiga** joylang. Ular `app.py` bilan yonma-yon turadi.
2. `backend/apply_backend_patch.py` faylini sinov tarmog‘iga qo‘shing. GitHub Codespaces terminali yoki mahalliy Python bilan repozitoriy ildizida `python3 apply_backend_patch.py .` buyrug‘ini bajaring. Skript `app.py` hamda `advanced_analysis.py` ni avtomatik tekshirib, backup yaratadi va ikki flag orqali yangi modullarni ulaydi.
3. Staging Render servisida environment variables:
   - `NOTA_LONG_AUDIO_ENABLED=true`
   - `NOTA_FULL_ALIGNMENT_ENABLED=true`
   - `NOTA_OMR_V3_ENABLED=true`
   - `NOTA_MAX_AUDIO_SECONDS=900`
   - `GEMINI_BACKUP_MODEL=` **faqat Google loyihangizda ruxsat etilgan, sinovdan o‘tgan model nomi**; bo‘sh qoldirish mumkin.
4. `GEMINI_API_KEY` va `NOTA_API_TOKEN` ni o‘zgartirmang, chatga yubormang.
5. GitHub staging commit saqlangandan keyin Render staging’da manual Deploy → Deploy latest commit ni bajaring (yoki ChatGPT’dan deployni boshlashni so‘rang). `main` uchun emas.
6. Test: real 3 daqiqa, 10 daqiqa, 15 daqiqa audio hamda foto PDF. Natijada `audio.duration_seconds`, `audio.total_detected_events`, `comparison.reference_count` va `comparison.performed_count` ni ko‘rib, hech qanday kesilish yo‘qligini tekshiring.
7. `comparison` faqat aniqlangan bir ovozli tovushlarning taxminiy mosligini beradi. Ritm/temp va pedagogik professional ball ishonchli dalilsiz chiqarilmaydi. Agar `Long alignment needs background batch processing` xatosi chiqsa, bu ongli cheklov — tizim natija uydirmaydi.

## Vercel marshrutlariga qo‘shish — FAQAT TO‘LIQ NEXT.JS SOURCE TOPILGANDA

GitHub’dagi `ohang-ai-video` repozitoriysi hozir to‘liq Next.js loyihasi emas. `vercel/apply_vercel_patch.py` ni **u yerda ishga tushirmang**. Uni faqat asl Next.js kodining mavjud `src/app/api/nota-render/route.js`, `src/app/api/credit-analysis/route.js`, `src/app/api/score-compare/route.js` marshrutlari yonida ishlating:

```
python3 vercel/apply_vercel_patch.py .
npm test
npm run build
```

Skript mavjud kod belgilarini tekshiradi, avvalgi fayllarga tegishdan oldin hammasi mosligini tekshiradi, keyin backup yaratadi. `maxDuration` 300s bo‘lishi deploy platforma plan/Fluid Compute cheklovlariga bog‘liq. Boshqa ichki `AbortController` timeoutlari va 96 MB fayl yuklash bundan alohida. Hozirgi sinxron endpoint 15 daqiqa davom etadigan bir so‘rovga yetarli emas; ishonchli yechim: **private object storage + persistent queue/worker + status polling + idempotent kredit hisoblash**. Shu arxitekturani kredit tizimi kodi to‘liq ko‘rilmasdan production’da qo‘shmang.

## Rollback

- Backend: `app.py.nota-backup` va `advanced_analysis.py.nota-backup` fayllarini qaytaring; eski versiyaga qaytish uchun `NOTA_LONG_AUDIO_ENABLED=false`, `NOTA_FULL_ALIGNMENT_ENABLED=false` qiling.
- Vercel: `route.js.nota-backup` fayllarini qaytaring; production loyiha o‘zgartirilmagan.
- Render’da eski muvaffaqiyatli deployni tiklash mumkin; production’ga chiqarishdan avval kredit va login regression sinovlari majburiy.

## Ehtiyotkorlik

- Ayni paytda 28 avtomatik test **haqiqiy Gemini endpointi orqali yakuniy muvaffaqiyatli transkripsiya** bo‘lganini bildirmaydi; AI javoblari nazorat testlarida imitasiya qilindi.
- OMR 52 soniyali vaqt limitida 48 sahifani qayta ishlay olmaydi; ko‘p sahifali nota uchun barqaror background worker kerak.
- Vercel source to‘liq tiklanmaganligi sabab patchni saytga avtomatik chiqarib bo‘lmadi.
