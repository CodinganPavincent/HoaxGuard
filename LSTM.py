import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import re

# DATASET
df0 = pd.read_excel("sample_data/cleaned/dataset_turnbackhoax_10_cleaned.xlsx")
df1 = pd.read_excel("sample_data/cleaned/dataset_cnn_10k_cleaned.xlsx")
df2 = pd.read_excel("sample_data/cleaned/dataset_kompas_4k_cleaned.xlsx")
df3 = pd.read_excel("sample_data/cleaned/dataset_tempo_6k_cleaned.xlsx")

df1.head()

# ## add summarized data from the same dataset. 
dfs0 = pd.read_excel("sample_data/summarized/dataset_turnbackhoax_summarized.xlsx")
# dfs1 = pd.read_excel("/kaggle/input/indonesian-fact-and-hoax-political-news/Summarized/dataset_cnn_summarized.xlsx")
# dfs2 = pd.read_excel("/kaggle/input/indonesian-fact-and-hoax-political-news/Summarized/dataset_kompas_summarized.xlsx")
# dfs3 = pd.read_excel("/kaggle/input/indonesian-fact-and-hoax-political-news/Summarized/dataset_tempo_summarized.xlsx")

# dfs1["tokens"] = dfs1.summarized.apply(lambda x: len(x.split()))
# dfs2["tokens"] = dfs2.summarized.apply(lambda x: len(str(x).split()))
# dfs3["tokens"] = dfs3.summarized.apply(lambda x: len(str(x).split()))


# CLEANING
dfn = pd.read_csv("sample_data/data.csv")

print(dfn.head())

# kesehatan key word
kes = 'kesehatan covid virus penyakit sakit kanker obat sembuh dokter rumah bakteri mati meninggal dunia parah'.split()
dfn["kes"] = dfn.content.apply(lambda x: len([k for k in str(x).lower().split() if k in kes]))

# politics key words 
pol = "jokowi, mk, mpr, dpr, golkar, tni, aparat, kementerian, polri, ditjen, uno, capres, survei, kpk, korupsi, mahfud, mulyani, ganjar, partai, pejabat, kpu, ruu, pemprov, dprd, heru, bahuri, kementrian, luhut, bupati, koalisi, dki, presiden, anies, pdip, firli, politik, thohir, gubernur".split(", ")
dfn["politik"] = dfn.content.apply(lambda x: len([k for k in str(x).lower().split() if k in pol]))

dfn["stoken"] = dfn.summary.apply(lambda x: len(str(x).split()))
dfn["ctoken"] = dfn.content.apply(lambda x: len(str(x).split()))

# okay dude. 
plt.hist(dfn.stoken)

# contain artis, tech, islam, science, kesehatan, covid
# dfh[dfh.politik==0].alls.values[100:200]

# plt.figure()
# plt.hist(dfs1.tokens)

# plt.figure()
# plt.hist(dfs2.tokens)

# plt.figure()
# plt.hist(dfs3.tokens)

sus1 = "Video, Foto, Akun, Artikel, Pesan, Surat, Gambar, Artikel, WhatsApp, Whatsapp, Hoax, Layar, Tangkapan, Klarifikasi, Tautan, Media".lower().split(", ")
sus2 = "narasi, Facebook, foto, video, Penjelasan, artikel, berita, media, Akun, Twitter, Video, Foto, gambar, Narasi, postingan, kategori, Konten, unggahan, Artikel, Beredar, cuitan, kutipan, dibagikan, hoaks, Penjelasan, penjelasan:, Menurut, dilansir".lower().split(", ")
sus3 = "informasi berdasarkan melalui penjelasan: pesan penelusuran media klaim beredar diunggah mengunggah postingan berantai whatsapp ditemukan menyebutkan dilansir situs dibagikan diklaim hoaks melalui fakta diambil".split()
sus = sus1 + sus2 + sus3

df = dfs0.copy()

df["alls"] = df["title"].astype(str).str.split(r"\]|\)", expand=True).drop(columns=0).stack().groupby(level=0).agg(lambda x: " ".join(x.dropna().astype(str)))

df["alls"] = df["alls"] + " " + df["cleaned"] 
df["alls"] = df.alls.apply(lambda x: " ".join([k for k in str(x).split() if k.lower() not in sus]))

df["alls"] = df["alls"].str.split("referensi|Referensi|REFERENSI|sumber|Sumber|SUMBER", expand=True)[0]

nword = df["alls"].apply(lambda x: len(str(x).split()))
nword.sort_values()

nt = df["title"].apply(lambda x: len(str(x).split()))
nt.sort_values()

n1 = df1.text_new.apply(lambda x: len(str(x).split()))
n1.sort_values()

n2 = df2.text_new.apply(lambda x: len(str(x).split()))
n2.sort_values()

n3 = df3.text_new.apply(lambda x: len(str(x).split()))
n3.sort_values()

nt.sort_values()
nword.sort_values()

plt.figure()
plt.hist(nword[(nword<1000)&(nword>50)])

plt.figure()
plt.hist(n1[n1<400])

plt.figure()
plt.hist(n2[n2<400])

plt.figure()
plt.hist(n3[n3<400])

# so small dude.
np.mean(nword[nword<400]), np.median(nword[nword<400]), np.sum((nword>40)&(nword<600))

# okay, using sample and index 0.15, 200, 150, 20

l = []
tot= 0
for k in [n1, n2, n3]:
    
    a1 = k[(k>20)&(k<100)].sample(frac=0.15, random_state=0).index.to_list()
    a2 = k[(k>100)&(k<200)].sample(200, random_state=0).index.to_list()
    a3 = k[(k>200)&(k<400)].sample(150, random_state=0).index.to_list()
    a4 = k[(k>400)&(k<600)].sample(20, random_state=0).index.to_list()
    
    l.append(a1+a2+a3+a4)
    tot+=len(a1+a2+a3+a4)

print(tot)

dfh = df.loc[nword[(nword<600)&(nword>30)].index,]
df_hoax = dfh.loc[dfh.politik>=0, "alls"]
ldf = len(df_hoax)
print(ldf)

dfk = pd.concat([df_hoax, df1.loc[l[0], "text_new"], df2.loc[l[1], 'text_new'], df3.loc[l[2],"text_new"], 
                 dfn.loc[(dfn.stoken>30)&(dfn.politik==0)&(dfn.kes>0),"summary"].sample(1000, random_state=0)
                ])
dfk.reset_index(drop=True, inplace=True)
dfk = pd.DataFrame(dfk)
dfk.columns = ['text']

dfk["label"] = [1]*ldf + [0]*(len(dfk)-ldf)
dfk.label.value_counts()

## adding some data
new_row = {"text": ["Pencegahan Penyakit yang dipicu oleh cacing ini dapat dicegah dengan memasak pada suhu yang melebihi 60 °C atau dengan membekukannya. Food and Drug Administration (FDA) menyarankan agar semua ikan dan kerang yang dimakan mentah dibekukan pada suhu −35 °C atau lebih rendah selama 15 jam atau dibekukan pada suhu −20 °C atau lebih rendah selama tujuh hari. Marinasi dan penggaraman tidak cukup untuk membunuh parasit ini. Di Uni Eropa, terdapat regulasi yang mewajibkan semua ikan yang akan dimakan mentah untuk dibekukan sebelum dijual. Regulasi ini berhasil memberantas penyakit anisakiasis di Belanda. Sebagai catatan, parasit Anisakis lebih sering ditemukan pada ikan di laut daripada ikan hasil budidaya.",
                    "bus yang mengangkut Calon Jamaah Haji asal Indonesia yang mengalami kecelakaan dan terbakar serta merenggut korban jiwa diantaranya dari Calon Jamaah Haji Kloter Blok Agung. Saat perjalanan dari Madinah-Mekkah bus yg penumpangnya dari kloter BLOK AGUNG diantaranya mas IMAM MAWARDI/ ISTRI dalam kondisi TERBAKAR sopirnya juga dari banyuwangi H.Syafei tewas..namun semua penumpang selamat..mari kita doakan semoga erjalanan selanjutnya aman dan lancar…aamiin"], 
           "label": [0, 1]}
dfk = pd.concat([dfk, pd.DataFrame(new_row)], ignore_index=True)
dfk["ntoken"] = dfk.text.apply(lambda x: len(str(x).split()))

dfk.head(), dfk.shape, dfk.label.value_counts()

plt.hist(dfk.loc[dfk.label==1, "ntoken"])
plt.hist(dfk.loc[dfk.label==0, "ntoken"])

dfk.to_csv('clean.csv', index=False, encoding="utf-8")