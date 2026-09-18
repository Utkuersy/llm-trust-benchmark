# Metodoloji ve kaynaklar

Bu belge, platformdaki her metodolojik seçimin dayanağını listeler. Amaç,
"bu eşiği neden 0.65 seçtin" türü sorulara izlenebilir cevap verebilmektir.

Aşağıdaki kaynaklar **yöntemi** gerekçelendirir. Hiçbiri bu platform için
sayısal ağırlık *reçete etmez*; sayılar, belirtilen yöntemlerin bu projenin
boyutlarına uygulanmasıyla türetilmiştir. Bu bir kalibrasyon değildir —
gerçek kalibrasyon için insan uzman etiketli referans set gerekir.

---

## 1. Ağırlık türetme

| Karar | Dayanak |
|---|---|
| Üst seviye boyutlar arasında eşit ağırlık | Dawes, R. M. (1979). *The robust beauty of improper linear models in decision making.* American Psychologist, 34(7), 571-582. Kalibrasyon verisi yokken eşit (birim) ağırlıkların, tahmin edilen ağırlıklara kıyasla dayanıklı sonuç verdiğini gösterir. |
| Boyut kümesinin ISO karakteristiklerine eşlenmesi | ISO/IEC 25010:2023, *Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model.* Standart 9 kalite karakteristiği tanımlar (functional suitability, performance efficiency, compatibility, interaction capability, reliability, security, maintainability, flexibility, safety). Bu 9 karakteristikten projeyle doğrudan ilgili 3 tanesi (Safety, Security, Functional suitability) seçilmiştir; standart bunlar arasında bir öncelik veya alt küme tanımlamaz, seçim ve aralarındaki eşit ağırlıklandırma bu projenin kararıdır. |
| Sıralı riskten sayısal ağırlığa geçiş (rank-sum) | Barron, F. H. & Barrett, B. E. (1996). *Decision Quality Using Ranked Attribute Weights.* Management Science, 42(11), 1515-1523. |
| Güvenlik alt boyutlarının sıralaması | OWASP Foundation (2024). *OWASP Top 10 for Large Language Model Applications 2025* (v2.0, 18 Kasım 2024). LLM01 Prompt Injection, LLM02 Sensitive Information Disclosure, LLM04 Data and Model Poisoning. |
| LLM güvenilirlik boyutlarının ayrıştırılması | Sun, L. vd. (2024). *TrustLLM: Trustworthiness in Large Language Models.* ICML 2024. |

**Uyarı.** ISO 25010 ve OWASP sayısal ağırlık vermez. "Ağırlıkları ölçtüm"
demek yanlıştır; doğru ifade "ağırlıkları belgelenmiş bir yöntemle
türettim"dir.

---

## 2. İçerik güvenliği testleri

| Karar | Dayanak |
|---|---|
| İşlevsel test paketi yaklaşımı (tek davranış = tek test) | Röttger, P. vd. (2021). *HateCheck: Functional Tests for Hate Speech Detection Models.* ACL-IJCNLP 2021, 41-58. DOI: 10.18653/v1/2021.acl-long.4. 29 işlevsellik, 3.728 doğrulanmış vaka. |
| İhlal olmayan karşıt vakaların zorunluluğu | Aynı kaynak. Yalnızca ihlal örnekleriyle test etmek yanlış pozitif oranını görünmez kılar. |
| Kaçırma tekniklerinin normalize edilmesi (yanlış yazım, harf arası noktalama, harf tekrarı) | Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. (2017). *Deceiving Google's Perspective API Built for Detecting Toxic Comments.* arXiv:1702.08138. "idiot" → "idiiot" değişikliğinin toksisite skorunu %84'ten %20'ye düşürdüğünü gösterir. |
| Olumsuzlama duyarlılığının bilinen sınırlama olarak belgelenmesi | Aynı kaynak; olumsuzlanmış küfürlü ifadelerin düşük skor almadığını raporlar. |
| Türkçe saldırgan dil için referans derlem | Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on Social Media.* LREC 2020. OffensEval 2020 Türkçe alt görevinin veri kaynağı. |
| `config/lexicons/{profanity,sexual}.txt` başlangıç kök seti | LDNOOBW (*List of Dirty, Naughty, Obscene, and Otherwise Bad Words*), 75 dilli açık kaynak topluluk listesi, Türkçe alt liste. github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words. **Kurumsal onaylı nihai liste değildir**, bkz. dosya başlıklarındaki not. |
| `profanity` (hedefsiz kaba dil) / `insult` (hedefli bireysel aşağılama) / `threat` (şiddet tehdidi) ayrımı | Aynı kaynak (Çöltekin, 2020) — OffensEval hiyerarşik etiketleme şeması (offensive/not → targeted/untargeted → hedef tipi). |

**Model seçimi notu.** Perspective API bu projede kullanılamaz: dış ağa
istek gönderir ve iç ağ kısıtına aykırıdır. Sınıflandırıcı katmanı yerel
diskten yüklenir (`content_safety.classifier_model_path`).

---

## 3. Dayanıklılık ve güvenlik testleri

| Karar | Dayanak |
|---|---|
| Düzenli ifadelerde süre sınırı testi | CWE-1333: *Inefficient Regular Expression Complexity.* OWASP: *Regular expression Denial of Service (ReDoS).* |
| Girdi boyutu sınırlaması | CWE-400: *Uncontrolled Resource Consumption.* OWASP ASVS v4.0.3, V5.1 Input Validation Requirements. |
| Log alanlarından kontrol karakteri temizliği | CWE-117: *Improper Output Neutralization for Logs.* |
| Yol argümanı doğrulaması | CWE-22: *Improper Limitation of a Pathname to a Restricted Directory.* |
| Alt süreçlerde `shell=False` | CWE-78: *OS Command Injection.* |
| Raporlarda hassas verinin maskelenmesi | Değerlendirme çıktısı paylaşılabilir bir artefakttır; ham PII'nin rapora yazılması sızıntıyı ölçmek yerine çoğaltır. |

---

## 4. Kimlik ve hesap numarası doğrulama

| Karar | Dayanak |
|---|---|
| Kart numarası kontrol hanesi (Luhn) | ISO/IEC 7812-1, *Identification cards — Identification of issuers.* |
| IBAN mod-97 kontrolü | ISO 13616-1, *Financial services — International Bank Account Number (IBAN).* |
| T.C. Kimlik Numarası kontrol hanesi | 10. ve 11. hane algoritması. |

Gerekçe: salt desen eşleştirme yanlış pozitif üretir. 11 haneli her sayı
kimlik numarası, 16 haneli her sayı kart numarası değildir.

---

## 5. Matematik değerlendirmesi

| Karar | Dayanak |
|---|---|
| Cevabın serbest metinden çıkarılması ve normalize edilerek karşılaştırılması | Hendrycks, D. vd. (2021). *Measuring Mathematical Problem Solving With the MATH Dataset.* NeurIPS Datasets and Benchmarks Track. |
| Soru kategorileri (Prealgebra, Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, Precalculus) | Aynı kaynak (Hendrycks vd., 2021) — MATH veri setinin resmi 7 konu taksonomisi. `data/math_eval/problems.jsonl` bu 7 kategorinin her birinden en az bir soru içerir; içerik projeye özel yazılmıştır, veri setinden kopyalanmamıştır. |
| Sembolik denklik kontrolü | Lewkowycz, A. vd. (2022). *Solving Quantitative Reasoning Problems with Language Models* (Minerva). NeurIPS 2022. Cevap denkliğinin string eşitliğiyle ölçülemeyeceğini ortaya koyar. |
| Çıkarılamayan cevabın yanlış sayılmaması | Format uyumsuzluğu ile muhakeme hatası farklı aksiyonlar gerektirir; ayrı raporlanır (`extraction_failures`). |

---

## 6. Yönetişim çerçeveleri

| Çerçeve | Bu projedeki karşılığı |
|---|---|
| NIST AI Risk Management Framework 1.0 (2023) | Measure fonksiyonu — sistem, risk ölçüm döngüsünün "measure" adımını uygular. |
| ISO/IEC 42001:2023 (AI yönetim sistemi) | Değerlendirme kayıtları ve sürüm izlenebilirliği (`SCORING_VERSION`, run kayıtları). |
| OWASP Top 10 for LLM Applications 2025 | Güvenlik boyutlarının taksonomisi ve sıralaması. |

---

## 7. Kalibrasyon durumu — açıkça belirtilmelidir

Şu an **kalibre edilmemiştir.** Sistem sıralama yapar (A modeli B'den
güvenli mi), mutlak eşik koymaz (70 puan üretime uygun mu). Kalibrasyon
için gerekenler:

1. En az 100 örnekten oluşan, iki bağımsız etiketleyicili altın set
2. Etiketleyiciler arası uyum (Cohen's kappa) raporu
3. Sistem çıktısının insan etiketiyle precision/recall karşılaştırması
4. Eşik seçiminin ROC eğrisi üzerinden gerekçelendirilmesi

Bunlar tamamlanana kadar sistem **karşılaştırma aracıdır, sertifikasyon
aracı değildir.**
