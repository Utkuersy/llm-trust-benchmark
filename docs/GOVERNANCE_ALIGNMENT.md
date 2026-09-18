# Yönetişim çerçeveleriyle hizalanma

Bu belge, platformun ölçtüğü şeyleri uluslararası AI yönetişim
çerçevelerine eşler. Amaç **yasal zorunluluk iddia etmek değil** —
kurumun gönüllü olarak tanınmış iyi yönetişim standartlarına hizalandığını
göstermektir. Bu ayrım, belgenin her yerinde korunur.

---

## 0. Önce bir düzeltme — neden "EU AI Act uyumu" yanlış çerçeveleme

İlk taslakta bu belge "EU AI Act'e uyum" olarak kurgulanmıştı. İki gerekçeyle
bu çerçeveleme terk edildi:

**Madde 2(3) — kapsam dışı bırakma.** EU AI Act'in 2. maddesinin 3. fıkrası,
yalnızca askeri, savunma veya milli güvenlik amacıyla kullanılan AI
sistemlerini **kapsam dışı** bırakır — "yüksek riskli" kategoriye değil,
regülasyonun kendisine tabi değildir. Recital 24, sistem sivil veya
dual-use amaçla da kullanılıyorsa kapsama geri girdiğini belirtir; ama
saf savunma kullanımı için muafiyet açıktır.

> Kaynak: Regulation (EU) 2024/1689, Article 2(3); Recital 24.
> "AI systems ... used exclusively for military, defence or national
> security purposes ... this Regulation does not apply."

**Yargı yetkisi.** AI Act, AB pazarına sürülmeyen veya çıktısı AB'de
kullanılmayan sistemleri bağlamaz (Madde 2). Şirket AB üyesi olmayan bir
ülkede kurulu ve sistem yalnızca iç ağda, iç kullanıcılar için çalışıyorsa,
regülasyonun *doğrudan* uygulanabilirliği tartışmaya açıktır.

**Sonuç.** "Bu sistem AI Act'in yüksek riskli kategorisine giriyor, o yüzden
Madde 15'e uymak zorundayız" cümlesi, bir AI Governance ekibinin önünde
kolayca çürütülebilir. Bunun yerine doğru ve savunulabilir cümle şudur:

> **"Yasal zorunluluk olmasa da, uluslararası kabul görmüş yönetişim
> çerçevelerinin (NIST AI RMF, ISO/IEC 42001) pratiklerini gönüllü olarak
> uyguluyoruz; EU AI Act'in teknik gereksinimlerini de — bağlayıcı olmasa
> bile — iyi tasarım referansı olarak kullanıyoruz. AB pazarına açık bir
> ürün/hizmet varsa bu eşleme hazır bir başlangıç noktası olur."**

Bu ikinci konumlanma daha güçlüdür, çünkü (a) doğrulanabilir bir iddiadır,
(b) gönüllü uyumun kendisi olgunluk göstergesidir, (c) yasal iddia
çürütüldüğünde projenin geri kalanının güvenilirliği sarsılmaz.

**Nihai karar hukuk/uyum biriminindir.** Bu belge bir hukuki görüş değildir;
şirketin belirli bir ürününün/hizmetinin AI Act kapsamına girip girmediği
sorusu nitelikli hukuk danışmanlığı gerektirir.

---

## 1. Birincil çapa: NIST AI Risk Management Framework 1.0 (2023)

Gönüllü, ABD merkezli ama coğrafyadan bağımsız kullanılan, dört fonksiyonlu
bir çerçeve: **Govern, Map, Measure, Manage.**

| NIST fonksiyonu | Platformdaki karşılığı |
|---|---|
| **Govern** — risk yönetimi kültürü ve süreçleri kurumsallaştırılmış mı | Ağırlık ön ayarlarının versiyonlanması (`SCORING_VERSION`), metodoloji belgesi, denetim izi (bkz. §4) |
| **Map** — bağlam ve risk kaynakları tanımlanmış mı | Yedi boyutun OWASP LLM Top 10 ve ISO 25010'a eşlenmesi (`docs/METHODOLOGY.md`) |
| **Measure** — riskler ölçülüyor mu, ölçüm güvenilir mi | Trust Score hesaplama, 135 testlik doğrulama paketi, insan kalibrasyon iskeleti (bkz. §5) |
| **Manage** — riskler önceliklendirilip azaltılıyor mu | Pipeline izleme (hangi aşamada risk oluştuğu), en riskli bulguların dashboard'da öne çıkarılması |

**Kanıt konumu.** `docs/METHODOLOGY.md`, `docs/DEFENSE.md`, `core/scoring.py`
(SCORING_VERSION sabiti), `core/trace.py`.

---

## 2. Birincil çapa: ISO/IEC 42001:2023 (AI Management System)

Gönüllü, uluslararası, ISO 9001 tarzı bir yönetim sistemi standardı —
"AI'ı nasıl sorumlu yönetiyorsunuz" sorusuna kurumsal süreç düzeyinde cevap
verir.

| ISO 42001 maddesi | Platformdaki karşılığı |
|---|---|
| 6.1 — risk ve fırsat değerlendirmesi | Yedi boyutlu risk taksonomisi |
| 8.1 — operasyonel planlama ve kontrol | `config/settings.yaml` merkezi konfigürasyon, hiçbir hardcoded eşik yok |
| 9.1 — izleme, ölçme, analiz, değerlendirme | Trust Score + pipeline izleme + drift raporu (bkz. §4) |
| 9.2 — iç denetim | Denetim izi tablosu (bkz. §4), 135 testlik doğrulama paketi |
| 10.1 — sürekli iyileştirme | Test paketinin dört gerçek hata bulup düzeltmesi; versiyonlanmış puanlama şeması |

---

## 3. İkincil ve şartlı referans: EU AI Act

Yalnızca aşağıdaki koşullardan biri gerçekleşirse **doğrudan bağlayıcı**
hale gelir; aksi halde **gönüllü iyi-tasarım referansı** olarak kullanılır:

- Sistem AB pazarına sürülür veya çıktısı AB'de kullanılır (Madde 2), **ve**
- Sistem yalnızca askeri/savunma/milli güvenlik amaçlı değildir, ya da
  sivil/dual-use bir kullanımı da vardır (Madde 2(3) istisnasının dışına
  çıkar).

Bu koşullar sağlansa bile sistemin Ek III'teki (Annex III) yüksek riskli
kategorilerden birine girip girmediği ayrıca değerlendirilmelidir — bu
platformun kapsamındaki bir iç RAG asistanı otomatik olarak "yüksek riskli"
sayılmaz.

**Koşullu olarak ilgili maddeler ve platform karşılığı:**

| Madde | Konu | Platformdaki karşılığı |
|---|---|---|
| Madde 9 | Risk yönetim sistemi | Yedi boyutlu risk taksonomisi, ağırlıklı puanlama |
| Madde 12 | Kayıt tutma (otomatik loglar) | JSON yapılandırılmış loglama, denetim izi tablosu |
| Madde 13 | Şeffaflık | `backend` alanının her sonuçta raporlanması (heuristic mi RAGAS mi kullanıldı gibi), METHODOLOGY.md |
| Madde 14 | İnsan gözetimi | İnsan kalibrasyon iskeleti (bkz. §5); dashboard'da bulguların insan incelemesine sunulması |
| **Madde 15** | **Doğruluk, sağlamlık, siber güvenlik** | **Doğrudan üçlü eşleme aşağıda** |

### Madde 15'in üç bileşeni ve platform karşılığı

Madde 15(1): *"...appropriate level of accuracy, robustness, and
cybersecurity, and that they perform consistently ... throughout their
lifecycle."*

| Madde 15 bileşeni | Platform boyutu |
|---|---|
| **Accuracy** (doğruluk) | `generation` (faithfulness/halüsinasyon), `math` (matematik doğruluğu), `retrieval` (doğru kaynağı bulma) |
| **Robustness** (sağlamlık) | `poisoning` (veri zehirlenmesi direnci), dayanıklılık test paketi (ReDoS, kaynak tüketimi, düşmanca girdi) |
| **Cybersecurity** (siber güvenlik) | `injection` (prompt injection direnci — Madde 15(5)'in özellikle bahsettiği "yetkisiz üçüncü taraf müdahalesi") |
| **Consistency throughout lifecycle** (yaşam döngüsü boyunca tutarlılık) | Drift raporu (bkz. §4) — model veya konfigürasyon değiştiğinde puan değişimini izler |

Bu tablo, Madde 15'in soyut gereksinimlerini somut, ölçülen metriklere
bağlar. "Yasal olarak buna tabiyiz" değil, **"Madde 15'in tarif ettiği üç
özelliği zaten ölçüyoruz, bağlayıcı olsun olmasın"** cümlesiyle sunulur.

---

## 4. Denetim izi ve tekrarlanabilirlik

Bkz. `core/audit.py` ve `core/versioning.py`. Her koşu şu bilgilerle
değiştirilemez biçimde (hash zinciriyle) kaydedilir:

- Kim tetikledi (işletim sistemi kullanıcısı, hostname)
- Ne zaman (UTC zaman damgası)
- Hangi kod sürümü (git commit hash, varsa)
- Hangi konfigürasyon (settings.yaml'ın hash'i)
- Hangi test verisi sürümü (korpus + senaryo + sözlük dosyalarının birleşik hash'i)

Bu, NIST RMF'nin Govern fonksiyonu ve ISO 42001'in 9.2 maddesinin
karşılığıdır: "bu sonucu nasıl elde ettin" sorusuna kanıt sunar.

---

## 5. İnsan kalibrasyonu — dürüstlük notu

`core/calibration.py`, bir **kalibrasyon iskeletidir** — örnekleme,
etiketleme şablonu üretme ve korelasyon hesaplama araçları hazır. Ancak
**gerçek bir insan etiketleme çalışması bu oturumda yapılmamıştır** çünkü
gerçek insan değerlendiriciler bu ortamda mevcut değildir.

Bu net söylenmelidir: *"Kalibrasyon altyapısı hazır ve test edilmiştir; asıl
çalışma insan değerlendiricilerle kurum tarafından yürütülmelidir."*
Sahte bir korelasyon sayısı üretip gerçekmiş gibi sunmak, bu belgenin
savunduğu dürüstlük ilkesinin tam tersidir ve tespit edildiğinde tüm
projenin güvenilirliğini yok eder.

---

## 6. Kaynaklar

1. Regulation (EU) 2024/1689 (EU AI Act) — Article 2(3), Article 9, Article
   12, Article 13, Article 14, Article 15, Recital 24.
2. NIST AI Risk Management Framework (AI RMF 1.0), Ocak 2023.
3. ISO/IEC 42001:2023 — Information technology — Artificial intelligence —
   Management system.
4. OWASP Top 10 for LLM Applications 2025 (v2.0).
5. ISO/IEC 25010:2023 — Product quality model.

> **Uyarı.** Bu belge hukuki görüş değildir. Şirketin belirli bir
> ürününün/hizmetinin hangi çerçevelere hangi ölçüde tabi olduğu, nitelikli
> hukuk ve uyum danışmanlığı gerektirir.
