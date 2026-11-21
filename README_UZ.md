# NOTA AI — OMR Photo v3 (sinov uchun xavfsiz yangilanish)

Bu modul foydalanuvchi yuborgan **«Dilnoz»** nota surati ustida tekshirildi. Kod mavjud Render FastAPI repositorysi (`dentresi455-commits/nota-ai-analysis-`) dagi `app.py` uchun mo‘ljallangan. Bu **production deploy emas**; GitHub orqali yozish ruxsati 403 sabab bloklangan.

## Qanday kuchaytirildi?
- Fotosurat orientatsiyasini besh chiziqli nota geometrik belgilariga qarab tekshirish. Misol suratda 90° burishni topdi. Yuqori/pastki 180° ikkiyoqlamalilik alohida ogohlantiriladi.
- Mayin deskew va CLAHE yordamida noaniq yoritilgan varaqlarning kontrastini yaxshilash.
- Muqobil qatlamda ko‘k rangdagi qo‘l yozuvlarini ajratish, **original qora notaga zarar yetkazmasdan** AI qayta tekshiruviga yordam berish.
- Nota satrlarini aniqlash. «Dilnoz» suratida **6 ta satr** topildi.
- Satrlar ichidagi takt va notalarni AI orqali JSON formatda o‘qish; dutor barmoq raqamlari **nota deb qabul qilinmasligi** haqida qat’iy ko‘rsatma.
- Asl va tozalangan ko‘rinishlarning ikki o‘qishi farqlansa, `omr_pass_disagreement` va `needs_review` qo‘yiladi.
- Satrlar va sahifalar ketma-ketligi, note-onset, davomiylik va takt offsetlarini birlashtirish.
- Tahlil natijalari **teacher-validated professional score** sifatida belgilab yuborilmaydi.

## O‘rnatish (faqat staging branch)
1. `photo_preprocessing.py` va `score_omr_v3.py` ni Render repo ildiziga ko‘chiring.
2. `requirements.txt` ga `opencv-python-headless>=4.10,<5` qo‘shing. Prod’ga emas, avval stagingda paket va xotira cheklovlarini tekshiring.
3. `python apply_to_render_app.py app.py` — eski reader qoladi va `app.py.omr-v3-backup` yaratiladi.
4. `NOTA_OMR_V3_ENABLED=true` — faqat staging; env yoqilmasa ilova **eski o‘qish tizimida qoladi**.
5. `python -m unittest -v test_score_omr_v2.py test_photo_omr_v3.py`.
6. Real kalit va autentifikatsiya bilan PDF+audio end-to-end sinovni faqat staging serverda o'tkazing.

## Hozircha ishlab turgan saytga kiritilmagani sabablar
- GitHub connector yangi branch yaratishga uringanda `403 Resource not accessible by integration` qaytardi.
- Vercel `credit-analysis → nota-render` oqimining Render microservice bilan birlashtirilgani ham alohida end-to-end tasdiqlanishi kerak.
- 48 sahifagacha bo‘lgan PDFni bitta HTTP so‘rov ichida ketma-ket o‘qish **Vercel/Render timeoutiga olib kelishi mumkin**. Uzun fayllar uchun alohida durable-job navbati, faylni saqlash, progress API va retry tizimi zarur. Shularsiz bu modulni katta asarlar uchun production-ready deb ko‘rsatish noto‘g‘ri.
- Real nota o‘qish (OMR) sifati boshqa qo‘lda tekshirilgan MusicXML/MIDI namunalarida o‘lchanmagan. Sintetik/mocked testlar aniqlik foizini isbotlamaydi.

## Fayllar
- `photo_preprocessing.py` — foto geometry, rangli annotation mask va staff segmentation
- `score_omr_v3.py` — Gemini bilan satrma-satr transkripsiya va ikki-pass tekshiruv
- `apply_to_render_app.py` — qaytariladigan, feature flag bilan ulash
- `test_photo_omr_v3.py`, `test_score_omr_v2.py` — lokal sinovlar
- `dilnoz_oriented_preview.jpg`, `dilnoz_enhanced_preview.jpg`, `dilnoz_annotation_suppressed_preview.jpg` — sinov surati namunalarining ko‘rinishi
- `test_results.txt` — test protokoli

### Eslatma
Siz yuborgan suratda oddiy 5 chiziqli nota va uning ustida **barmoq raqamlari** bor. Raqamlarni nota deb o‘qish katta metodik xato. Tizim nota balandligini chiziqlardagi joylashuvga qarab aniqlashi kerak. Nota balandligi, temp/ritm, dutor sozlanishi va oktava masalalarini real audiodan mustaqil tekshirish zarur.

## Maxfiy surat bilan test
`qa_private/dilnoz_photo.jpeg` nomiga original suratingizni lokal kiriting;
bu papka `.gitignore`da va ZIP arxivda **ataylab yo‘q**. Sinovlarda mavjud bo‘lsa
haqiqiy suratning orientatsiya va satr deteksiyasi ham tekshiriladi.
Siz yuborgan notani tasodifan umumiy GitHub repositoryga joylashtirish tavsiya etilmaydi.
