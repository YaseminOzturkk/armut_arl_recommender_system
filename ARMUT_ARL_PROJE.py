
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


#########################
# GÖREV 1: Veriyi Hazırlama
#########################

# Adım 1: armut_data.csv dosyasınız okutunuz.


df_ = pd.read_csv("C:/Users/yasmi/Desktop/recommender_systems/ARL_Project_Armut/armut_data.csv")
df = df_.copy()
df.head
df.info()
df.describe().T
df.isnull().sum()
df.shape


# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["Services"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df.head()

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df.info()

# dt.to_period metodu,tarih veya zaman bilgisini dönemlere dönüştürmek için kullanılır.

df["NewDate"] = df["CreateDate"].dt.to_period("M")
df.head()

df["BasketId"] = df["UserId"].astype(str) + "_" + df["NewDate"].astype(str)
df.head()

#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..


# Önce BasketId'ye göre kır, sonra Services'lere göre, sepette satın alınmış Service'ten kaç tane var:
# CategoryId Services'in içinde bu nedenle Services'in count'unu da alabiliriz.


# unstack: groupby işleminden sonra pivot yapmak için yani "Services" değişkenini sütunlara geçirmek için unstack() fonsiyonunu kullanılır.
# Eksik değerleri 0, dolularda 1 yazsın istiyoruz.

# unstack() işleminden sonra boş olan yerleri 0 ile doldurmak için fillna(0) metodunu kullanılır.


# 0'dan büyük herhangi bir sayıya 1, diğerlerinde 0 yazmalıyız.
# Çünkü daha ölçülebilir üzerinde analitik işlemler yapabileceğimiz özel bir matris yapısı bekliyoruz.
# Burada applymap() fonksiyonunu kullanıyoruz. Çünkü, applymap() bütün gözlemleri gezer (satır ve sütunların hepsinde).

apriori_df = df.groupby(["BasketId", "Services"])["CategoryId"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)


# Adım 2: Birliktelik kurallarını oluşturunuz.

# İlk olarak apriori() fonksiyonu ile olası tüm ürün birlikteliklerinin Support değerlerini yani olasılıklarını bulalım.
# Burada min_support, belirlemek istediğimiz minimum Support değeri, threshold
# Kullanmak istediğimiz veri setindeki değişkenlerin isimlerini kullanmak istiyorsak use_colnames=True eklenir.

frequent_itemsets = apriori(apriori_df.astype("bool"),
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

# Şu anda elimizde olası Services veya Services çiftleri ve bunlara karşılık support değerleri verilmiş.
# Burada 0.01'in altındaki olası değerler yok çünkü minimum support değerini(eşik değeri) 0.01 olarak vermiştik.
# Bunlar her bir hizmetin olasılığıdır. Bizim ihtiyacımız olan birliktelik kurallarıdır. Dolayısıyla bu veriyi kullanıp
# bunun üzerinden birliktelik kurallarını çıkaracağız.



# İhtiyacımız olan birliktelik kuralları için association_rules() metodu ile
# bu veriyi kullanıp bunun üzerinden birliktelik kurallarını çıkaracağız:

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules.head()

# antecedents: Önceki Hizmet
# consequents: İkinci Hizmet
# antecedent support: İlk Hizmetin tek başına gözlenme olasılığı
# consequent support: İkinci Hizmetin tek başına gözlenme olasılığı
# support: İki Hizmetin birlikte görülme olasılığı
# confidence: İlk hizmet alındığında  ikinci Hizmetin alınma olasılığı
# lift: Bir Hizmet alındığında ikinci Hizmetin alınma olasılığının kaç kat artacağının belirtir.
# leverage: lift benzeridir. Support u yüksek olan değerlere öncelik verme eğilimindedir bundan dolayı ufak bir yanlılığı vardır.
# conviction: Bir Hizmet olmadan diğer Hizmetin beklenen frakansı

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, service_id, rec_num=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, service in enumerate(sorted_rules["antecedents"]):
        for j in list(service):
            if j == service_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_num]


# service_id: Öneri yapılmasını istediğimiz service id'si.
# rec_num: İstenen sayıda tavsiye hizmeti getirir.
# İlk olarak en uyumlu ilk ürünü yakalayabilmek için kuralları lifte göre büyükten kücüğe sıraladık.
# (Bu sıralama tercihe göre confidence'e göre de olabilir.)
# Tavsiye edilecek ürünler için boş bir liste oluşturulur.
# Sıralanmış kurallarda ilk önce gelen service göre enumerate() metodunu kullanıyoruz.
# İkinci döngüde service'lerde gezilecek. Eğer tavsiye istenen hizmet yakalanırsa,
# index bilgisi i ile tutuluyordu bu index bilgisindeki consequents değerini recommendation_list'e ekler.
# [0] ilk gördüğü hizmeti getirmesi için eklenir.

arl_recommender(rules,"2_0",3)
df.head

# Not: Önerilen hizmet sayısı arttıkça diğer denk gelen hizmetlerin ilgili istatistiklerdeki değerleri daha düşük olacaktır.