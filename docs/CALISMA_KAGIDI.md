# Çalışma Kağıdı
## LLM Çıktı Güvenilirliği Değerlendirme Platformu

**Hazırlayan:** [ad soyad] · **Tarih:** [tarih] · **Kapsam:** Staj projesi ara raporu

---

## 1. Bir paragrafta proje

Kurumsal bir RAG asistanının **çıktılarını** güvenilirlik açısından ölçen,
iç ağda çalışacak şekilde tasarlanmış bir değerlendirme platformu. Yedi
boyut ölçülür (içerik güvenliği, prompt injection direnci, PII sızıntısı,
veri zehirlenmesi, retrieval kalitesi, faithfulness, matematik yeteneği) ve
tek bir Trust Score'da (0-100) birleştirilir. Boyutlar ayrı ayrı da
raporlanır. Sistem ayrıca pipeline'ın hangi aşamasında risk oluştuğunu
gösterir. Ağırlıklar keyfi değildir; ISO/IEC 25010 ve OWASP LLM Top 10
üzerinden belgelenmiş bir yöntemle türetilmiştir.

---

## 2. Problem tanımı ve konumlandırma

### 2.1 Çözülen problem

Bir LLM asistanının **yanlış** cevap vermesi düzeltilebilir bir hatadır.
Ancak küfürlü, hakaret içeren, dini değerlere saldıran veya kişisel veri
sızdıran bir cevap üretmesi kurumsal bir olaydır. Bu ikinci risk sınıfı
çoğu değerlendirme kurulumunda ölçülmez.

İkinci problem: RAG bir zincirdir (soru → bağlam getirme → üretim → filtre).
Yalnızca son çıktıya bakıldığında sorunun *nerede* oluştuğu anlaşılmaz.
Kök neden analizi yapılamaz.

### 2.2 Mevcut çözümlerle ilişki — LLM-Stats

Görevlinin işaret ettiği [llm-stats.com/benchmarks](https://llm-stats.com/benchmarks),
680 benchmark ve 55 yetenek alanı üzerinden model sıralamaları üreten bir
**benchmark toplayıcısıdır**. Her kategorideki sonuçlar TrueSkill muhafazakâr
derecelendirmesiyle birleştirilir.

LLM-Stats'in kendi metodoloji notu, bu projenin gerekçesini doğrudan ifade
eder: sıralamalar prompt formatı, harness sürümü, kontaminasyon, eksik
koşular ve model güncellemeleriyle değişebilir; bu nedenle **tek bir
evrensel skor yerine birden çok ilgili test** kullanılmalıdır.

| | LLM-Stats | Bu platform |
|---|---|---|
| Ölçtüğü | Modelin genel yeteneği | Bizim pipeline'ımızın çıktısı |
| Verisi | Kamuya açık benchmark setleri | Kurumun kendi korpusu ve soruları |
| Dili | Ağırlıklı İngilizce | Türkçe biçimbilime uyarlanmış |
| Kapsamı | Reasoning, coding, math, vision… | İçerik güvenliği, PII, injection, RAG kalitesi |
| Çalıştığı yer | Kamuya açık, çevrimiçi | İç ağ / hava kapalı |
| Cevapladığı soru | "Hangi model daha iyi?" | "Bizim kurulumumuz üretime uygun mu?" |

**Sonuç:** Bunlar rakip değil tamamlayıcıdır. LLM-Stats model *seçimine*
yardım eder; bu platform seçilen modelin bizim koşullarımızda ne ürettiğini
ölçer. Kamuya açık bir benchmark, bir modelin iç dokümanları özetlerken
T.C. kimlik numarası sızdırıp sızdırmadığını söyleyemez.

**Matematik köprüdür.** Görevlinin özellikle işaret ettiği math kategorisi
LLM-Stats'te de indekslenir (MATH, AIME, MMLU-Pro aileleri). Bu platformun
matematik modülü, MATH benchmark ailesinin cevap çıkarma ve denklik kontrolü
yaklaşımını kullanır; böylece kurum içi ölçüm kamuya açık sıralamalarla aynı
metodolojik zemine oturur.

---

## 3. Mimari — yedi katman

Bağımlılık tek yönlüdür: üstteki katman alttakini kullanır, tersi olmaz.
Dairesel bağımlılık yoktur.

```
┌──────────────────────────────────────────────────────────┐
│  7. SUNUM            app.py (Streamlit + Plotly)         │
│                      yalnızca okur, koşu tetiklemez      │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  6. DEPOLAMA         SQLite · MLflow · results/*.json     │
│                      her koşu tekrar üretilebilir         │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  5. PUANLAMA         core/scoring.py                      │
│                      kaynağa bağlı ağırlık ön ayarları    │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  4. PIPELINE İZLEME  core/trace.py                        │
│     input_guardrail → retrieval → generation → output_gr. │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  3. ANALİZ (7 boyut, her biri bağımsız modül)             │
│     content_safety · injection · pii · poisoning          │
│     retrieval · faithfulness · math                       │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  2. ÇEKİRDEK         şemalar · JSON loglama · subprocess  │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  1. KONFİGÜRASYON    settings.yaml + Pydantic             │
│                      hiçbir modülde hardcoded değer yok   │
└──────────────────────────────────────────────────────────┘
```

### Katmanların gerekçesi

| Katman | Sorumluluk | Neden ayrı olmalı |
|---|---|---|
| 1 Konfigürasyon | Tüm eşik, ağırlık ve yollar | Ağırlık değişimi kod değişimi gerektirmez; ortam değişkeniyle ezilebilir |
| 2 Çekirdek | Şema, log, güvenli subprocess | Her analiz modülünün ortak ihtiyacı; bir kez yazılır |
| 3 Analiz | 7 boyut, 7 ayrı modül | Her modül tek başına çalıştırılabilir; biri bozulsa diğerleri çalışır |
| 4 Pipeline izleme | Aşama bazlı trace | Kök neden analizi; riskin hangi aşamadan geldiği |
| 5 Puanlama | Ağırlık ön ayarları + sürümleme | Ağırlık şeması versiyonlanır; eski skorlar karşılaştırılabilir kalır |
| 6 Depolama | SQLite + MLflow + ham JSON | İzlenebilirlik; her koşu tekrar üretilebilir |
| 7 Sunum | Dashboard | Sadece okur; ağır bağımlılık gerektirmez, ayrı dağıtılabilir |

**Bu ayrımın pratik karşılığı:** Kurum "içerik güvenliği ağırlığı %50 olsun"
derse tek satır YAML değişir. "Yeni injection senaryosu ekleyin" derse tek
dosya. "Chroma yerine Qdrant kullanın" derse tek sınıf. Hiçbiri diğerini
kırmaz.

---

## 4. Değerlendirme çerçevesi

### 4.1 Yedi boyut ve ağırlıkları

Boyutlar, ISO/IEC 25010:2023'ün tanımladığı 9 kalite karakteristiğinden
(functional suitability, performance efficiency, compatibility, interaction
capability, reliability, security, maintainability, flexibility, safety)
projeyle doğrudan ilgili 3 tanesine eşlenir: Safety, Security, Functional
suitability. Diğer 6 karakteristik (performans, uyumluluk, taşınabilirlik
vb.) bu değerlendirmenin kapsamı dışındadır.

| ISO sütunu | Boyut | Ne ölçer | Ağırlık |
|---|---|---|---|
| **Safety** | İçerik güvenliği | Küfür, hakaret, dini değerlere saldırı, tehdit, cinsel içerik | 0.33 |
| **Security** | Injection direnci | 12 saldırı senaryosunda savunmanın kırılma oranı | 0.17 |
| | PII sızıntısı | TCKN, IBAN, kart, API anahtarı, e-posta, telefon | 0.11 |
| | Zehirlenme direnci | Korpusa sokulan sahte dokümana kanma oranı | 0.06 |
| **Functional** | Retrieval | Context precision / recall, MRR | 0.11 |
| | Faithfulness | Halüsinasyon oranı, cevap alaka düzeyi | 0.11 |
| | Matematik | Cevap doğruluğu (sembolik denklik kontrolüyle) | 0.11 |

### 4.2 Ağırlıklar nasıl türetildi

**İki adımlı yöntem:**

**Adım 1 — üst seviye: eşit ağırlık.** ISO'nun 9 karakteristiğinden
seçilen 3 tanesi (Safety, Security, Functional suitability) arasında eşit
bölüşüm (her biri 1/3). Bu seçim ve eşitleme projenin kararıdır; standart
kendisi bu 3'ü ayrıca gruplamaz, sadece 9'u da aynı düzeyde tanımlar ve
aralarında öncelik vermez. Kalibrasyon verisi yokken eşit ağırlığın dayanıklı bir
seçim olduğu literatürde gösterilmiştir.

> Dawes, R. M. (1979). *The robust beauty of improper linear models in
> decision making.* American Psychologist, 34(7), 571-582.

**Adım 2 — güvenlik içi bölüşüm: rank-sum.** OWASP sıralaması sayısal
ağırlığa çevrilir: `w_i = (n + 1 − r_i) / Σ(n + 1 − r)` → 3 : 2 : 1.

> Barron, F. H. & Barrett, B. E. (1996). *Decision Quality Using Ranked
> Attribute Weights.* Management Science, 42(11), 1515-1523.
>
> OWASP Top 10 for LLM Applications 2025 (v2.0, 18 Kasım 2024):
> LLM01 Prompt Injection, LLM02 Sensitive Information Disclosure,
> LLM04 Data and Model Poisoning.

**Hesap:** `0.33 (Safety) + [0.17 + 0.11 + 0.06] (Security) + [0.11 × 3]
(Functional) = 1.00`

### 4.3 Kritik puanlama kuralı

Bir boyut ölçülemezse (sözlük boş, cevap dosyası yok) **sıfır puan
verilmez**; o boyutun ağırlığı paydadan düşülür ve kalan boyutlara orantılı
dağıtılır. Aksi halde altyapı eksikliği modele haksız ceza olarak yansırdı —
değerlendirme sistemlerinin sık yaptığı bir hata.

Ayrıca ölçülmeyen kategoriler raporda `inactive_categories` altında görünür.
**"İhlal bulunamadı" ile "aranmadı" asla karıştırılmaz.**

---

## 5. Ne yapıldı — hangi kaynağa dayanarak

### 5.1 İçerik filtresi kaçırma tekniklerine karşı normalizasyon

**Problem.** Sözlük tabanlı filtreler, kelimeyi yanlış yazarak veya harf
arasına noktalama koyarak atlatılır.

**Çözüm.** Tarama öncesi normalizasyon katmanı: Türkçe'ye duyarlı küçük
harfe çevirme, harf değiştirme çözümü (`@→a`, `1→i`), harf arası ayraç
birleştirme (`a.p.t.a.l → aptal`), üç ve üzeri harf tekrarını tek karaktere
indirme, Unicode NFKC. Ayrıca Türkçe eklemeli yapıya karşı kök + ek
toleranslı eşleştirme.

**Kaynak.**
> Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. (2017). *Deceiving
> Google's Perspective API Built for Detecting Toxic Comments.*
> arXiv:1702.08138.

Çalışma, küfürlü kelimelerin yanlış yazılması veya araya noktalama
eklenmesiyle sistemin atlatılabildiğini gösterir; "idiot" → "idiiot"
değişikliği aynı cümlenin toksisite oranını **%84'ten %20'ye** düşürmüştür.
Aynı çalışma, sistemin olumsuzlanmış küfürlü ifadelere düşük skor vermediğini
de raporlar.

---

### 5.2 İşlevsel test paketi ve karşıt vakalar

**Problem.** Bir içerik filtresini yalnızca ihlal örnekleriyle test etmek
yanıltıcıdır: her şeyi işaretleyen bir filtre de %100 başarılı görünür.

**Çözüm.** Her test tek bir davranışı sınar. Testlerin bir kısmı **ihlal
olmayan** metinlerdir (nötr kurumsal metin, nazik eleştiri, akademik bağlam)
ve işaretlenmemeleri beklenir.

**Kaynak.**
> Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., &
> Pierrehumbert, J. (2021). *HateCheck: Functional Tests for Hate Speech
> Detection Models.* ACL-IJCNLP 2021, 41-58.
> DOI: 10.18653/v1/2021.acl-long.4

Çalışma, önceki araştırmaların incelenmesi ve sivil toplum paydaşlarıyla
görüşmelerden 29 model işlevselliği tanımlar, test vakalarını yapılandırılmış
bir etiketleme süreciyle doğrular ve 29 işlevsel test altında **3.728 vaka**
yayımlar; her vaka nefret içerikli / içermeyen altın etiketi taşır. Bu
yöntemle test edilen hem akademik hem ticari modellerde kritik zayıflıklar
ortaya çıkmıştır.

---

### 5.3 Kimlik numarası doğrulama

**Problem.** Salt desen eşleştirmeyle 11 haneli her sayı TCKN, 16 haneli her
sayı kart numarası sayılır; sipariş numaraları yanlışlıkla PII işaretlenir.

**Çözüm.** Her sayısal kimlik kendi kontrol algoritmasından geçirilir;
doğrulamayı geçmeyen eşleşme raporlanmaz.

**Kaynaklar.** ISO/IEC 7812-1 (Luhn kontrol hanesi) · ISO 13616-1 (IBAN
mod-97) · T.C. Kimlik Numarası 10. ve 11. hane algoritması.

---

### 5.4 Matematik cevap denkliği

**Problem.** `1/2` ile `0.5`, `1,200` ile `1200`, `x^2-9` ile `x**2-9` aynı
cevaptır. Ham string karşılaştırma doğruluğu olduğundan düşük gösterir.

**Çözüm.** Üç kademeli kontrol: normalize string eşitliği → tolerans
dahilinde sayısal denklik → SymPy ile sembolik denklik. Cevap serbest
metinden çıkarılır (`\boxed{}`, "Cevap:", son satır kalıpları). Cevabı
çıkarılamayan sorular **yanlış sayılmaz**, ayrı raporlanır — format
uyumsuzluğu ile muhakeme hatası farklı aksiyon gerektirir.

**Kaynaklar.**
> Hendrycks, D. vd. (2021). *Measuring Mathematical Problem Solving With the
> MATH Dataset.* NeurIPS Datasets and Benchmarks Track.
>
> Lewkowycz, A. vd. (2022). *Solving Quantitative Reasoning Problems with
> Language Models* (Minerva). NeurIPS 2022.

---

### 5.5 Pipeline izleme

**Problem.** Cevap kötüyse sorumlunun retrieval mı, model mi, filtre mi
olduğu anlaşılmaz.

**Çözüm.** Her sorgu için dört aşama ayrı ölçülür; süre, durum, girdi/çıktı
özeti ve o aşamada tetiklenen guardrail bulguları kaydedilir.

**Somut kazanım.** "6 bulgu var" yerine **"6 bulgunun tamamı çıktı
aşamasında — risk kullanıcıdan gelmiyor, modelin kendisi PII sızdırıyor"**
denebiliyor. İki durum farklı aksiyon gerektirir: biri girdi filtresi,
diğeri model/prompt değişikliği.

**Dayanak.** Dağıtık sistemlerdeki span/trace modelinin RAG hattına
uyarlanması. OWASP LLM Top 10 2025'te LLM05 (Improper Output Handling) ve
LLM08 (Vector and Embedding Weaknesses) ayrı riskler olarak tanımlanır —
yani çıktı ve retrieval ayrı denetlenmelidir.

---

### 5.6 Platformun kendi güvenliği

**Problem.** Bu araç tanım gereği güvenilmeyen metin işler; değerlendirilen
modelin çıktısı saldırgan tarafından kontrol edilebilir.

**Çözüm.** Dört zayıflık sınıfı için otomatik test:

| Zayıflık | Kaynak | Önlem |
|---|---|---|
| ReDoS — düzenli ifadede üstel geri izleme | CWE-1333, OWASP ReDoS | Patolojik girdide süre sınırı testi |
| Kontrolsüz kaynak tüketimi | CWE-400, OWASP ASVS v4.0.3 V5.1 | Girdi boyutu kırpma |
| Log injection | CWE-117 | Bağlam alanında kontrol karakteri yasağı |
| Komut enjeksiyonu | CWE-78 | Hiçbir yerde `shell=True` yok |

Ek olarak raporlarda ham hassas veri taşınmaz, bulgular maskelenir. Gerekçe:
değerlendirme raporu paylaşılabilir bir artefakttır; ham PII'yi rapora yazmak
sızıntıyı ölçmek yerine çoğaltmak olurdu.

---

### 5.7 İç ağ / hava kapalı çalışma

**Problem.** Modern NLP kütüphaneleri çalışma anında model ağırlığı indirir;
hava kapalı ortamda bu sessiz hatalara yol açar.

**Çözüm.** Üç kademeli yedek zinciri — vektör deposu (Chroma → NumPy),
embedding (sentence-transformers → hashing), faithfulness (RAGAS →
heuristic). `offline.enforce` ayarı `HF_HUB_OFFLINE` ve
`TRANSFORMERS_OFFLINE` değişkenlerini uygular. Docker'da değerlendirme
servisi `network_mode: none` ile çalışır.

**Kritik nokta.** Hangi arka ucun kullanıldığı çıktıda `backend` alanıyla
raporlanır. Kalite sessizce düşmez, şeffaf şekilde etiketlenir.

---

## 6. Doğrulama

### 6.1 Test paketi

135 otomatik test, üç grup:

- **İşlevsel testler** — HateCheck yöntemi, karşıt vakalar dahil
- **Dayanıklılık testleri** — ReDoS, kaynak tüketimi, log injection,
  düşmanca girdi tipleri
- **Doğrulayıcı testleri** — Luhn, IBAN, TCKN, matematik denkliği, ağırlık
  normalizasyonu

### 6.2 Test paketinin bulduğu gerçek hatalar

| Hata | Etkisi |
|---|---|
| Harf tekrarı normalizasyonu yanlış çalışıyordu | Güvenlik filtresi bilinen bir kaçırma tekniğini atlatıyordu |
| PII bağlam alanında satır sonu kalıyordu | Log injection açığı (CWE-117) |
| Cevap çıkarmada kelime sınırı yoktu | "cevaplayamıyorum" içindeki "cevap" hecesi işaretçi sanılıyordu |
| Veritabanı alan eşlemesi hatalıydı | Pipeline verisi sessizce sıfır yazılıyordu |

Dördü de test yazılmasaydı sessizce yanlış sonuç üretmeye devam edecekti.

### 6.3 CI'da otomatik doğrulamalar

1. **Ayırt etme testi** — PII sızdıran ve injection'a yenilen model, temiz
   modellerden düşük puan almalı; almazsa build kırılır.
2. **Pipeline testi** — her modelde aşama izi kaydedilmiş olmalı.
3. **Ağırlık duyarlılık testi** — üç farklı ağırlık ön ayarında model
   sıralaması değişmemeli.

Üçüncüsü özellikle önemlidir: **ağırlık seçimi tartışmaya açıktır, ama
sonucun ona duyarlı olmadığı gösterilmiştir.** Bu, türetilmiş bir ağırlık
iddiasından farklı olarak *gerçek bir ölçüm* iddiasıdır.

---

## 7. Bilinen sınırlar

Bir ölçüm aracının en önemli özelliği neyi ölçemediğini bilmesidir.

1. **Kalibre edilmemiştir.** Sistem *sıralama* yapar, *mutlak eşik* koymaz.
   Kalibrasyon için en az 100 örneklik, iki bağımsız etiketleyicili altın set
   ve etiketleyiciler arası uyum (Cohen's kappa) raporu gerekir.
   **Karşılaştırma aracıdır, sertifikasyon aracı değildir.**
2. **Sözlük katmanı bağlam duyarlı değildir.** Terimin akademik veya alıntı
   bağlamında geçmesi ihlal değildir ama sözlük bunu ayırt edemez. Bu
   davranış `xfail` testleriyle belgelenmiştir, gizlenmemiştir.
3. **Heuristic faithfulness, LLM-as-judge değildir.** RAGAS bir LLM
   sağlayıcısı olmadan çalışmaz; yedek mod sayısal halüsinasyonları iyi
   yakalar, anlamsal çelişkileri kaçırabilir.
4. **Sözlükler henüz boştur.** Kurum doldurana kadar içerik güvenliği puanı
   "temiz" değil, kısmen "aranmadı" anlamına gelir.
5. **Erişim kontrolü yoktur.** Dashboard kimlik doğrulaması içermez ve
   raporlar hassas bulgular taşır.
6. **Bağımsız güvenlik denetimi yapılmamıştır.** Dayanıklılık testleri
   belirli zayıflık sınıflarını kapsar; sızma testi yerine geçmez.

---

## 7.5. Kurumsal olgunluk katmanı (staj görevlisi geri bildirimiyle eklendi)

Görevlinin talebi üzerine sistem "iyi çalışan bir araç" seviyesinden
"kurumun güvendiği referans" seviyesine taşındı:

| Eklenen | Ne çözüyor | Dosya |
|---|---|---|
| Yönetişim çerçeve eşlemesi | "Hangi standarda dayanıyor" sorusuna cevap | `docs/GOVERNANCE_ALIGNMENT.md` |
| Denetim izi | "Bu sonucu nasıl elde ettin, kanıtla" | `core/audit.py` |
| Versiyonlama + drift | "Puan neden değişti — veri mi, model mi" | `core/versioning.py` |
| İnsan kalibrasyonu iskeleti | "Bu sayı gerçekten anlamlı mı" (altyapı hazır, gerçek çalışma bekliyor) | `core/calibration.py` |
| Eklenti mimarisi | "Yeni boyut eklemek motoru bozar mı" — hayır, kanıtlandı | `core/dimensions.py` |
| Çok turlu saldırı testleri | Tek mesajlık taramanın kaçırdığı kademeli saldırılar | `llm_security/prompt_injection_tests.py` |

**Kritik düzeltme.** İlk planda "EU AI Act uyumu" en yüksek öncelikliydi.
Araştırma sonucu bu çerçeveleme değiştirildi: AI Act Madde 2(3) askeri/
savunma amaçlı sistemleri kapsam dışı bırakıyor, ayrıca yargı yetkisi sorusu
var. Bunun yerine NIST AI RMF ve ISO/IEC 42001 (gönüllü, coğrafyadan
bağımsız) birincil çapa yapıldı, EU AI Act "bağlayıcı olmasa da iyi
tasarım referansı" olarak ikincil konuma alındı. Bu ayrımı görevline
mutlaka anlat — yanlış hukuki iddiada bulunmak, hiç iddia etmemekten
kötüdür.

## 8. Sonraki adımlar (öncelik sırasıyla)

1. **Sözlüklerin doldurulması.** Kurumdan yazılı kılavuz: ihlal sayılan
   20-30 örnek **ve** ihlal sayılmayan 20-30 sınır örnek. İkinci liste
   olmadan yanlış pozitif oranı ölçülemez.
2. **Kalibrasyon.** 100 örneklik altın set, iki etiketleyici, kappa raporu,
   sistem çıktısının insan etiketiyle precision/recall karşılaştırması.
3. **Kimlik doğrulama.** Dashboard iç ağda yayına alınmadan önce.
4. **Yerel sınıflandırıcı.** Sözlüğün yakalayamadığı örtük toksisite için;
   model ağırlıkları önceden diske indirilerek.

---

## 9. Muhtemel sorular ve cevapları

**"Ağırlıkları nasıl belirledin?"**
İki adımlı türetme: ISO/IEC 25010 karakteristikleri arasında eşit ağırlık
(Dawes 1979), güvenlik içinde OWASP LLM Top 10 sıralamasına rank-sum
(Barron & Barrett 1996). **Ölçmedim, türettim** — ölçmek için sonuç
değişkeni ve regresyon gerekir, öyle bir veri yok.

**"Ağırlıklar farklı olsa sonuç değişir miydi?"**
Mutlak puanlar değişir, sıralama değişmez. Üç ön ayarla test edildi, CI'da
otomatik doğrulanıyor.

**"Bu sistem güvenli mi, iç ağda çalıştırabilir miyiz?"**
Dört zayıflık sınıfına karşı otomatik test var ve geçiyor. Ancak bağımsız
sızma testi yapılmadı ve dashboard'da kimlik doğrulama yok. "Test edilmiş"
diyebilirim, "denetlenmiş" diyemem.

**"İçerik güvenliği 100 çıkıyor, sistem çalışıyor mu?"**
Sözlükler henüz boş; bu puan büyük ölçüde "aranmadı" demek. Dashboard bunu
sarı uyarı olarak gösteriyor. Gerçek liste girilmeden bu boyut anlamlı
sonuç üretmez.

**"LLM-Stats varken buna neden ihtiyacımız var?"**
LLM-Stats modelin genel yeteneğini ölçer, bizim pipeline'ımızın çıktısını
değil. Kamuya açık bir benchmark, bir modelin bizim iç dokümanlarımızı
özetlerken kimlik numarası sızdırıp sızdırmadığını söyleyemez. Ayrıca
LLM-Stats'in kendi metodolojisi de tek evrensel skor yerine birden çok
ilgili test öneriyor.

**"Neden kod analizi hattını kaldırdın?"**
Kurumun problemi AI'ın yazdığı kod değil, asistanın kullanıcıya ne
söylediği. Yarısı kullanılmayan bir sistem sunmak yerine tek hatta
odaklandım.

---

## 10. Kaynakça

**İçeriği doğrudan doğrulanmış olanlar:**

1. Röttger, P. vd. (2021). *HateCheck: Functional Tests for Hate Speech
   Detection Models.* ACL-IJCNLP 2021, 41-58. DOI 10.18653/v1/2021.acl-long.4
2. Hosseini, H. vd. (2017). *Deceiving Google's Perspective API Built for
   Detecting Toxic Comments.* arXiv:1702.08138
3. OWASP Foundation (2024). *OWASP Top 10 for Large Language Model
   Applications 2025* (v2.0, 18 Kasım 2024)
4. llm-stats.com/benchmarks — kapsam ve metodoloji notu

**Standart referanslar** (sunumda kullanmadan önce künyeyi teyit et):

5. ISO/IEC 25010:2023 — Yazılım kalite modeli (9 karakteristik; bu
   projede ilgili 3'ü kullanılmıştır)
6. ISO/IEC 7812-1 — Kart numarası tanımlama (Luhn)
7. ISO 13616-1 — IBAN
8. CWE-1333, CWE-400, CWE-117, CWE-78 — Zayıflık sınıfları
9. OWASP ASVS v4.0.3 — Girdi doğrulama gereksinimleri
10. Dawes, R. M. (1979). American Psychologist, 34(7), 571-582
11. Barron, F. H. & Barrett, B. E. (1996). Management Science, 42(11), 1515-1523
12. Hendrycks, D. vd. (2021). MATH Dataset. NeurIPS D&B
13. Lewkowycz, A. vd. (2022). Minerva. NeurIPS
14. Sun, L. vd. (2024). TrustLLM. ICML
15. Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on Social
    Media.* LREC 2020

> **Uyarı:** Bir kaynağı sunumda kullanacaksan önce aç ve oku. "Şu makaleye
> göre yaptım" deyip makale sorulduğunda cevap verememek, hiç kaynak
> göstermemekten daha kötüdür.
