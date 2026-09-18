# İçerik güvenliği sözlükleri

Her `.txt` dosyası bir kategoriyi temsil eder; dosya adı kategori adıdır.
Kategorinin önem derecesi `config/settings.yaml` içindeki
`content_safety.category_severity` bölümünde tanımlanır.

## Format

- Satır başına bir **terim kökü** (ek almadan önceki hâli)
- `#` ile başlayan satırlar yorumdur
- Kökler en az 4 karakter olmalıdır (kısa kökler yanlış pozitif üretir)
- Türkçe ekler otomatik tolere edilir: `aptal` kökü `aptalsın`, `aptallar`
  gibi türevleri de yakalar (kök + en fazla 6 karakter ek)
- Harf değiştirme (a→@, i→1), harf arası ayraç (a.p.t.a.l) ve harf tekrarı
  (aptaaal) tarayıcı tarafından normalize edilir; bunların varyantlarını
  ayrıca yazmaya gerek yoktur

## Boş dosyalar

Boş bırakılan bir kategori **pasif** sayılır ve tarama sonucunda
`inactive_categories` altında raporlanır. Sistem hata vermez, ama o
kategori ölçülmemiş olur — raporda bu açıkça görünür.

## Kurumsal onay gerektiren kategoriler

`religious_insult.txt` ve benzeri kültürel bağlam taşıyan kategorilerin
içeriği **kurum tarafından tanımlanmalıdır**. Dine yönelik akademik veya
eleştirel bir ifade ile hakaret arasındaki sınır teknik bir karar değildir.
Bu dosya doldurulmadan önce, kabul edilebilir ve edilemez örneklerden
oluşan yazılı bir kılavuz alınması önerilir.
