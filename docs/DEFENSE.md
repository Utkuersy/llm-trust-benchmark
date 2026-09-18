# Savunma notları — her kararın dayanağı

Bu belge tek bir soruya cevap verir: **"Bunu neden böyle yaptın?"**

Her satır bir problem, ona verilen çözüm ve çözümün dayandığı kaynağı içerir.
Sunumda bu belgeyi açık tutabilir, sorulan her karara buradan cevap
verebilirsin.

---

## 0. Projenin konumu: LLM-Stats ne yapar, bu platform ne yapar

Görevlinin gösterdiği [llm-stats.com/benchmarks](https://llm-stats.com/benchmarks),
680 benchmark ve 55 yetenek alanı üzerinden model sıralamaları üreten bir
**benchmark toplayıcısıdır**. Her yetenek kategorisindeki benchmark sonuçları
TrueSkill muhafazakâr derecelendirmesiyle birleştirilip tek bir indeks
hâline getirilir.

LLM-Stats'in kendi metodoloji notu, bu platformun varlık gerekçesini
doğrudan ifade eder: sıralamalar prompt formatı, harness sürümü,
kontaminasyon, eksik koşular ve model güncellemeleriyle değişebilir; bu
nedenle **tek bir evrensel skor yerine birden çok ilgili test**
kullanılmalıdır.

Aradaki fark şudur:

| | LLM-Stats | Bu platform |
|---|---|---|
| Ölçülen | Modelin **genel yeteneği** | Bizim pipeline'ımızın **çıktısı** |
| Veri | Kamuya açık benchmark setleri | Kurumun kendi korpusu ve soruları |
| Dil | Ağırlıklı İngilizce | Türkçe biçimbilime uyarlanmış |
| Kapsam | Reasoning, coding, math, vision… | İçerik güvenliği, PII, injection, RAG kalitesi |
| Çalışma yeri | Kamuya açık, çevrimiçi | İç ağ / hava kapalı |
| Cevapladığı soru | "Hangi model daha iyi?" | "Bizim kurulumumuz üretime uygun mu?" |

**Bunlar rakip değil, tamamlayıcıdır.** LLM-Stats model seçimine yardım eder;
bu platform seçilen modelin *bizim korpusumuzla, bizim promptumuzla, bizim
kullanıcılarımıza* ne ürettiğini ölçer. Kamuya açık bir benchmark, bir
modelin kurumun iç dokümanlarını özetlerken kişisel veri sızdırıp
sızdırmadığını söyleyemez.

**Matematik boyutu köprüdür.** Görevlinin özellikle işaret ettiği math
kategorisi, LLM-Stats'in de indekslediği bir yetenek alanıdır. Bu platformun
`capability/math_eval.py` modülü, MATH benchmark ailesinin cevap çıkarma ve
denklik kontrolü yaklaşımını kullanır — yani kurum içi ölçüm, kamuya açık
sıralamalarla aynı metodolojik zemine oturur ve karşılaştırılabilir olur.

---

## 1. Problem → çözüm → kaynak eşlemesi

### 1.1 "Tek bir skor yanıltıcıdır"

**Problem.** Model değerlendirmesinde yaygın hata, her şeyi tek bir sayıya
indirgemektir. Yüksek genel skorlu bir model, kurumun asıl önemsediği
boyutta (içerik güvenliği) kötü olabilir.

**Çözüm.** Yedi ayrı boyut ölçülür, her biri kendi alt puanını korur;
birleşik Trust Score bunların ağırlıklı ortalamasıdır ve alt puanlar her
zaman ayrı raporlanır. Dashboard önce boyutları, sonra toplamı gösterir.

**Dayanak.** LLM-Stats metodoloji notu: tek evrensel skor yerine birden çok
ilgili test. Ayrıca TrustLLM (Sun vd., ICML 2024) güvenilirliği tek sayı
değil, ayrı ayrı raporlanan boyutlar olarak tanımlar.

---

### 1.2 "Ağırlıklar keyfi görünüyor"

**Problem.** Boyutları birleştirmek ağırlık gerektirir. "İçerik güvenliği
neden %33?" sorusuna cevap verilemezse tüm skor tartışmalı hâle gelir.

**Çözüm — iki katmanlı türetme.**
1. ISO/IEC 25010:2023, 9 kalite karakteristiği tanımlar (functional suitability,
   performance efficiency, compatibility, interaction capability, reliability,
   security, maintainability, flexibility, safety). Bunlardan projeyle
   doğrudan ilgili 3 tanesi seçildi — Safety, Security, Functional
   suitability — ve **aralarında eşit ağırlık** verildi. Bu seçim ve eşit
   ağırlıklandırma standardın kendisinin değil, bu projenin kararıdır.
2. Security içindeki bölüşüm, OWASP LLM Top 10 sıralamasına **rank-sum**
   yöntemi uygulanarak yapılır: `w_i = (n+1-r_i) / Σ(n+1-r)`.

**Dayanak.**
- ISO/IEC 25010:2023 — 9 karakteristiklik tam liste; bu projede ilgili 3
  tanesi seçildi. Standart, seçilmemiş 6 karakteristik (performance
  efficiency, compatibility, interaction capability, reliability,
  maintainability, flexibility) arasında da bir öncelik sıralaması vermez —
  bu yüzden seçilen 3'e eşit ağırlık verilmesi standardın ruhuna uygundur,
  ama standardın doğrudan emrettiği bir şey değildir.
- Dawes, R. M. (1979), *The robust beauty of improper linear models in
  decision making*, American Psychologist 34(7) — kalibrasyon verisi
  yokken eşit ağırlıkların dayanıklı olduğunu gösterir.
- Barron & Barrett (1996), *Decision Quality Using Ranked Attribute
  Weights*, Management Science 42(11) — sıralı tercihten ağırlık türetme.
- OWASP Top 10 for LLM Applications 2025 (v2.0, 18 Kasım 2024) — LLM01
  Prompt Injection, LLM02 Sensitive Information Disclosure, LLM04 Data and
  Model Poisoning sıralaması.

**Söylenecek cümle.** *"Ağırlıkları ölçmedim, belgelenmiş bir yöntemle
türettim. Bu bir kalibrasyon değil, gerekçelendirilmiş başlangıç noktası."*

**Ölçtüğüm şey ise şu:** ağırlık seçiminin sonuca etkisi. Üç farklı ön
ayarla (`output_safety_first`, `owasp_rank`, `trustllm_equal`) koşuldu ve
model sıralaması değişmedi. Bu test CI'da çalışır; sıralama değişirse build
kırılır. **Bu gerçek bir ölçüm iddiasıdır.**

---

### 1.3 "İçerik filtresi kolayca atlatılır"

**Problem.** Sözlük tabanlı filtreler, kelimeyi yanlış yazarak veya harf
arasına noktalama koyarak atlatılır. Bu teorik bir risk değil, belgelenmiş
bir saldırıdır.

**Çözüm — normalizasyon katmanı.** Tarama öncesi metin normalize edilir:
Türkçe'ye duyarlı küçük harfe çevirme, harf değiştirme çözümü (`@→a`,
`1→i`), harf arası ayraç birleştirme (`a.p.t.a.l → aptal`), üç ve üzeri
harf tekrarını tek karaktere indirme (`aptaaaal → aptal`), Unicode NFKC
normalizasyonu.

**Dayanak.** Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. (2017),
*Deceiving Google's Perspective API Built for Detecting Toxic Comments*,
arXiv:1702.08138. Çalışma, küfürlü kelimelerin yanlış yazılması veya araya
noktalama eklenmesiyle sistemin atlatılabildiğini gösterir; "idiot"
kelimesini "idiiot" yapmak aynı cümlenin toksisite oranını %84'ten %20'ye
düşürmüştür.

**Kanıt.** Bu davranış `tests/test_content_safety.py::test_f2_*` altında
altı ayrı kaçırma tekniğiyle test edilir. Test paketi yazıldığında harf
tekrarı normalizasyonunda gerçek bir hata bulundu ve düzeltildi — sistem
`aptaaaal` girdisini kaçırıyordu.

---

### 1.4 "Filtrenin yanlış pozitif oranı bilinmiyor"

**Problem.** Bir içerik filtresini yalnızca ihlal örnekleriyle test etmek
yanıltıcıdır: her şeyi işaretleyen bir filtre de %100 başarılı görünür.
Yanlış pozitif oranı ölçülmeden filtre üretime alınamaz.

**Çözüm — işlevsel test paketi + karşıt vakalar.** Her test tek bir
davranışı sınar. Testlerin bir kısmı **ihlal olmayan** metinlerdir (nötr
kurumsal metin, nazik eleştiri, akademik bağlam) ve bunların
işaretlenmemesi beklenir.

**Dayanak.** Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H.,
& Pierrehumbert, J. (2021), *HateCheck: Functional Tests for Hate Speech
Detection Models*, ACL-IJCNLP 2021, 41-58, DOI 10.18653/v1/2021.acl-long.4.
Çalışma önceki araştırmaların incelenmesi ve sivil toplum paydaşlarıyla
görüşmelerden 29 model işlevselliği tanımlar, test vakalarını yapılandırılmış
bir etiketleme süreciyle doğrular ve 29 işlevsel test altında 3.728 vaka
yayımlar; her vaka nefret içerikli / içermeyen altın etiketi taşır. Bu
yöntemle test edilen hem akademik hem ticari modellerde kritik zayıflıklar
ortaya çıkmıştır.

**Dürüstlük notu.** İki karşıt vaka şu an **başarısız** ve `xfail` olarak
işaretli: "Aptallık üzerine bir psikoloji makalesi okudum" cümlesi yanlış
pozitif üretiyor. Bu gizlenmedi — sözlük katmanının bağlam duyarlı olmadığı
mimari bir sınırdır ve belgelenmiştir. Test `strict=True` ile durur: ileride
sınıflandırıcı eklenip davranış düzelirse test uyarı verir.

---

### 1.5 "Kimlik numarası tespiti yanlış pozitif üretir"

**Problem.** Salt desen eşleştirmeyle 11 haneli her sayı TCKN, 16 haneli her
sayı kart numarası sayılır. Sipariş numaraları, ürün kodları ve tarihler
yanlışlıkla PII olarak işaretlenir.

**Çözüm — yapısal doğrulama.** Her sayısal kimlik tipi kendi kontrol
algoritmasından geçirilir; doğrulamayı geçmeyen eşleşme raporlanmaz.

**Dayanak.**
- ISO/IEC 7812-1 — kart numaraları için Luhn kontrol hanesi
- ISO 13616-1 — IBAN mod-97 kontrolü
- T.C. Kimlik Numarası 10. ve 11. hane algoritması

**Kanıt.** `tests/test_validators_and_scoring.py` içinde geçerli ve geçersiz
örneklerle test edilir. Ayrı bir test, rastgele 11 haneli sayıların
çoğunun reddedildiğini doğrular — doğrulayıcının varlık gerekçesi budur.

---

### 1.6 "Matematik cevabı doğru ama sistem yanlış sayıyor"

**Problem.** `1/2` ile `0.5`, `1,200` ile `1200`, `x^2-9` ile `x**2-9` aynı
cevaptır. Ham string karşılaştırma bunları yanlış sayar ve doğruluk oranını
olduğundan düşük gösterir.

**Çözüm — üç kademeli denklik kontrolü.** Önce normalize edilmiş string
eşitliği, sonra tolerans dahilinde sayısal denklik, sonra SymPy ile sembolik
denklik. Ayrıca cevap serbest metinden çıkarılır (`\boxed{}`, "Cevap:",
son satır kalıpları).

**Dayanak.**
- Hendrycks, D. vd. (2021), *Measuring Mathematical Problem Solving With
  the MATH Dataset*, NeurIPS Datasets & Benchmarks — cevap normalizasyonu
  ve denklik kontrolü gerekliliği
- Lewkowycz, A. vd. (2022), *Solving Quantitative Reasoning Problems with
  Language Models* (Minerva), NeurIPS 2022

**Ek karar.** Cevabı **çıkarılamayan** sorular yanlış sayılmaz, ayrı
raporlanır (`extraction_failures`). Gerekçe: format uyumsuzluğu ile muhakeme
hatası farklı aksiyonlar gerektirir — biri prompt düzeltmesi, diğeri model
değişikliği.

**Kanıt.** Test paketi bu katmanda da gerçek bir hata buldu: `extract_answer`
fonksiyonu "cevaplayamıyorum" kelimesinin içindeki "cevap" hecesini işaretçi
sanıp `layamıyorum.` döndürüyordu. Kelime sınırı ve zorunlu ayraç eklendi.

---

### 1.7 "Cevap kötü ama nerede bozulduğu belli değil"

**Problem.** RAG bir zincirdir. Sadece son çıktıya bakınca sorumlunun
retrieval mı, model mi, yoksa filtre mi olduğu anlaşılmaz. Kök neden analizi
yapılamaz.

**Çözüm — pipeline izleme katmanı.** Her sorgu için dört aşama ayrı ölçülür:
`input_guardrail → retrieval → generation → output_guardrail`. Her aşamanın
süresi, durumu, girdi/çıktı özeti ve o aşamada tetiklenen guardrail bulguları
kaydedilir.

**Somut kazanım.** "6 bulgu var" demek yerine **"6 bulgunun tamamı çıktı
aşamasında — risk kullanıcıdan gelmiyor, modelin kendisi PII sızdırıyor"**
denebiliyor. Bu iki durum farklı aksiyon gerektirir: biri girdi filtresi,
diğeri model veya prompt değişikliği.

**Dayanak.** Bu, LLM gözlemlenebilirliğinin (observability) standart
yaklaşımıdır; dağıtık sistemlerdeki span/trace modelinin RAG hattına
uyarlanmış hâlidir. OWASP LLM Top 10 2025'te LLM05 (Improper Output
Handling) ve LLM08 (Vector and Embedding Weaknesses) ayrı riskler olarak
tanımlanır — yani çıktı ve retrieval ayrı denetlenmelidir.

---

### 1.8 "Bu araç kendisi bir saldırı yüzeyi"

**Problem.** Platform, tanım gereği güvenilmeyen metin işler:
değerlendirilen modelin çıktısı saldırgan tarafından kontrol edilebilir.
Tarayıcının kendisi hedef olabilir.

**Çözüm — dayanıklılık test paketi.** Dört zayıflık sınıfı test edilir:

| Zayıflık | Kaynak | Test |
|---|---|---|
| ReDoS — düzenli ifadede üstel geri izleme | CWE-1333; OWASP ReDoS | Patolojik girdide süre sınırı |
| Kontrolsüz kaynak tüketimi | CWE-400; OWASP ASVS v4.0.3 V5.1 | Girdi boyutu kırpma |
| Log injection | CWE-117 | Bağlam alanında kontrol karakteri yasağı |
| Komut enjeksiyonu | CWE-78 | Hiçbir yerde `shell=True` yok |

**Ek karar.** Raporlarda ham hassas veri taşınmaz; bulgular maskelenir
(`meh***@***le`). Gerekçe: değerlendirme raporu paylaşılabilir bir
artefakttır; ham PII'yi rapora yazmak sızıntıyı ölçmek yerine çoğaltmak
olurdu.

---

### 1.9 "Ölçülmeyen boyut, temiz boyut gibi görünüyor"

**Problem.** Bir sözlük boş bırakılmışsa hiçbir ihlal bulunmaz ve boyut
100 puan alır. Bu, olmayan bir güvence verir — değerlendirme araçlarının en
tehlikeli hatası.

**Çözüm.** Ölçülmeyen kategoriler `inactive_categories` altında raporlanır,
dashboard'da sarı uyarı olarak gösterilir. Ayrıca `skipped` durumundaki bir
boyutun ağırlığı paydadan düşülür ve kalan boyutlara dağıtılır — sıfır puan
verilmez, boyut hesaptan çıkarılır.

**Söylenecek cümle.** *"'İhlal bulunamadı' ile 'aranmadı' aynı şey değildir;
sistem bu ikisini asla karıştırmaz."*

---

### 1.10 "İç ağda çalışamaz, model indirmeye çalışır"

**Problem.** Modern NLP kütüphaneleri çalışma anında model ağırlığı indirir.
Hava kapalı ortamda bu sessiz hatalara yol açar.

**Çözüm — üç katmanlı fallback + offline zorlaması.**
- Vektör deposu: Chroma → FAISS → saf NumPy
- Embedding: sentence-transformers → hashing embedding
- Faithfulness: RAGAS → heuristic

`offline.enforce: true` ayarı `HF_HUB_OFFLINE` ve `TRANSFORMERS_OFFLINE`
değişkenlerini süreç geneline uygular. Sınıflandırıcı modeli yerel diskten
yüklenir. Docker'da değerlendirme servisi `network_mode: none` ile çalışır.

**Kritik nokta.** Hangi arka ucun kullanıldığı çıktıda `backend` alanıyla
raporlanır. Kalite sessizce düşmez, şeffaf şekilde etiketlenir.

---

## 2. Katman mimarisi — "kaç katman ve neden"

Yedi katman, bağımlılık tek yönlü akar (üstteki alttakini import eder,
tersi olmaz):

| # | Katman | Sorumluluk | Neden ayrı |
|---|---|---|---|
| 1 | **Konfigürasyon** | `settings.yaml` + Pydantic | Hiçbir modülde hardcoded eşik/ağırlık yok; ortam değişkeniyle ezilebilir |
| 2 | **Çekirdek altyapı** | Şemalar, JSON loglama, güvenli subprocess | Her analiz modülünün ortak ihtiyacı; bir kez yazılır |
| 3 | **Analiz** | 7 boyut, her biri ayrı modül | Her modül tek başına çalıştırılabilir; biri bozulsa diğerleri çalışır |
| 4 | **Pipeline izleme** | Aşama bazlı trace | Kök neden analizi; hangi aşamada bozulduğu |
| 5 | **Puanlama** | Kaynağa bağlı ağırlık ön ayarları | Ağırlık değişikliği kod değişikliği gerektirmez; sürüm etiketlenir |
| 6 | **Depolama ve izlenebilirlik** | SQLite + MLflow + JSON | Her koşu tekrar üretilebilir; puanlama sürümü kaydedilir |
| 7 | **Sunum** | Streamlit dashboard | Sadece okur; ağır bağımlılık gerektirmez |

**Neden bu ayrım önemli.** Kurum yarın "içerik güvenliği ağırlığı %50 olsun"
derse tek satır YAML değişir. "Yeni bir injection senaryosu ekleyin" derse
tek dosya değişir. "Chroma yerine Qdrant kullanın" derse tek sınıf eklenir.
Hiçbiri diğerini kırmaz.

---

## 3. Kanıtlanabilir iddialar (sunumda güvenle söyleyebileceklerin)

1. **"Test paketi dört gerçek hata buldu ve düzeltti."** Biri güvenlik
   filtresini atlatan kaçırma tekniği, biri log injection vektörü, biri
   cevap çıkarma hatası, biri veritabanına sıfır yazan alan eşleme hatası.
2. **"Sonuç ağırlık seçimine duyarlı değil."** Üç farklı ön ayarda model
   sıralaması aynı; bu CI'da otomatik doğrulanıyor.
3. **"Sistem güvensiz örneği ayırt ediyor."** PII sızdıran ve injection'a
   yenilen model, temiz modellerden belirgin düşük puan alıyor; CI'da
   assertion olarak duruyor.
4. **"Pipeline'ın hangi aşamasında risk oluştuğunu söyleyebiliyorum."**
   Aşama bazlı bulgu dağılımı ve darboğaz tespiti kaydediliyor.
5. **"135 otomatik test, 2 belgelenmiş sınırlama."** Sınırlamalar gizlenmiyor,
   `xfail` ile işaretli ve gerekçeli.

---

## 4. Kanıtlanamayan iddialar (asla söyleme)

1. ❌ *"Ağırlıkları ölçtüm."* → Türettim. Ölçmek için sonuç değişkeni ve
   regresyon gerekir; öyle bir veri yok.
2. ❌ *"Sistem kalibre edildi."* → Edilmedi. Kalibrasyon için en az 100
   örneklik, iki bağımsız etiketleyicili altın set ve Cohen's kappa raporu
   gerekir.
3. ❌ *"İç ağda çalışacak kadar güvenli."* → Dayanıklılık testleri belirli
   zayıflık sınıflarını kapsar; bağımsız sızma testi yerine geçmez. Ayrıca
   dashboard'da kimlik doğrulama yok.
4. ❌ *"İçerik güvenliği %100."* → Sözlükler kurum tarafından doldurulmadığı
   sürece bu sayı "aranmadı" anlamına gelir.

**Bu dört sınırı kendin söyle, sorulmasını bekleme.** Sınırını bilen bir
ölçüm aracı, bilmeyenden daha güvenilirdir.

---

## 5. Kaynak listesi

Bu oturumda içeriği doğrudan doğrulananlar:

- Röttger vd. (2021), HateCheck, ACL-IJCNLP 2021, DOI 10.18653/v1/2021.acl-long.4
- Hosseini vd. (2017), Deceiving Google's Perspective API, arXiv:1702.08138
- OWASP Top 10 for LLM Applications 2025 (v2.0, 18 Kasım 2024)
- llm-stats.com/benchmarks — metodoloji ve kapsam

Standart referanslar (kullanmadan önce künyeyi kendin teyit et):

- ISO/IEC 25010:2023 — yazılım kalite modeli
- ISO/IEC 7812-1 — Luhn kontrol hanesi
- ISO 13616-1 — IBAN
- CWE-1333, CWE-400, CWE-117, CWE-78 — zayıflık sınıfları
- OWASP ASVS v4.0.3 — girdi doğrulama gereksinimleri
- Dawes (1979), American Psychologist 34(7), 571-582
- Barron & Barrett (1996), Management Science 42(11), 1515-1523
- Hendrycks vd. (2021), MATH Dataset, NeurIPS D&B
- Lewkowycz vd. (2022), Minerva, NeurIPS
- Sun vd. (2024), TrustLLM, ICML
- Çöltekin (2020), A Corpus of Turkish Offensive Language, LREC

**Uyarı.** Bir kaynağı sunumda kullanacaksan önce aç ve oku. "Şu makaleye
göre yaptım" deyip makale sorulduğunda cevap verememek, hiç kaynak
göstermemekten daha kötüdür.
