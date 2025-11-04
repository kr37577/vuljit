import re

def extract_c_cpp_project_names(data):
    """
    プロジェクトデータの文字列を解析し、CまたはC++で書かれたプロジェクトの名前を抽出します。

    この関数は、2番目の列に言語が記載されていることを前提としています。
    (例: 'gpac c 4 ...')

    Args:
        data: プロジェクトデータを含む複数行の文字列。

    Returns:
        CまたはC++プロジェクトの名前（文字列）のリスト。
    """
    # 条件に一致するプロジェクト名を保持するリスト
    project_names = []
    
    # 生のデータ文字列を個々の行に分割
    lines = data.strip().split('\n')

    # テーブルのヘッダー（参考用）
    if lines:
        header = lines[0]
        print(f"ヘッダーを検出: {header}\n")
    
    # ヘッダーをスキップして、データの各行を反復処理
    for line in lines[1:]:
        # 空白行はスキップ
        if not line.strip():
            continue

        # 行を空白文字で分割します。これにより、列間の複数のスペースやタブにも対応できます。
        parts = re.split(r'\s+', line.strip())
        
        # パーツが少なくとも2つあることを確認（プロジェクト名と言語）
        if len(parts) < 2:
            continue

        # 2番目の部分（言語）をチェック
        language = parts[1]

        # 言語が 'c' または 'c++' かどうかを確認
        if language == 'c' or language == 'c++':
            # プロジェクト名（1番目の部分）をリストに追加
            project_names.append(parts[0])

    return project_names

# ユーザーから提供された新しい形式のデータ
data = """
Project name	Language	Fuzz target count	Runtime code coverage	Total lines	Lines covered
zydis	c	3	73.09%	12171	8896
zip	c	2	36.30%	6973	2531
xs	c	2	73.44%	66332	48716
xnu	c	1	14.20%	255674	36311
xen	c	1	81.72%	9784	7995
wolfmqtt	c	1	4.42%	53952	2387
wazuh	c	1			
wasm3	c	1	65.66%	4167	2736
wamr	c	6			
w3m	c	1	70.38%	6400	4504
vulkan-loader	c	6	16.11%	25543	4116
vlc	c	27	35.18%	124711	43873
varnish	c	1	38.34%	2285	876
util-linux	c	4	27.40%	35895	9836
utf8proc	c	1	92.24%	670	618
unit	c	5	19.16%	33402	6400
unbound	c	5	8.83%	67046	5922
tmux	c	1	14.01%	52662	7380
tinysparql	c	4	14.98%	363073	54389
tdengine	c	1			
tarantool	c	30	21.39%	367914	78685
sudoers	c	8	61.15%	20920	12792
spdk	c	1	66.00%	750	495
sound-open-firmware	c	7			
selinux	c	5	62.30%	70025	43624
samba	c	190	33.19%	913385	303147
ruby	c	1	0.00%	219952	0
rtpproxy	c	4	34.99%	30733	10755
rauc	c	2	6.85%	326365	22354
rabbitmq-c	c	3	8.71%	5422	472
quickjs	c	3	37.48%	51552	19323
qemu	c	1			
python3-libraries	c	8	40.03%	263403	105443
pycryptodome	c	8	80.33%	1398	1123
pupnp	c	1	48.98%	4112	2014
protobuf-c	c	1	49.12%	3367	1654
proftpd	c	1	1.30%	59238	771
postgresql	c	2	1.49%	579673	8643
postfix	c	2			
plan9port	c	1	16.92%	5816	984
pjsip	c	13	27.79%	30259	8408
pidgin	c	2	7.02%	194256	13633
pacemaker	c	17			
p11-kit	c	1	6.63%	20432	1354
ostree	c	2	5.41%	170592	9235
oss-fuzz-example	c	2	71.64%	134	96
opusfile	c	1	73.13%	2851	2085
openvpn	c	10	10.93%	46489	5082
opensips	c	4	13.41%	67958	9116
open5gs	c	2	25.25%	17980	4540
oniguruma	c	1	89.94%	26513	23846
numactl	c	1	38.45%	2091	804
ntpsec	c	3			
nokogiri	c	1			
nginx	c	1	13.42%	93147	12499
msquic	c	2	27.36%	196495	53764
ms-tpm-20-ref	c	1	55.83%	21052	11754
mpv	c	20	50.23%	71433	35880
mosquitto	c	20	30.45%	26527	8078
miniz	c	9	60.27%	5575	3360
memcached	c	1			
mdbtools	c	1	32.72%	4799	1570
md4c	c	1	97.52%	4319	4212
lxc	c	3	16.26%	34887	5672
lua	c	16	30.15%	89698	27041
llhttp	c	1	74.63%	9389	7007
lldpd	c	4	7.53%	18190	1369
lighttpd	c	1	35.39%	1280	453
libyang	c	3	37.68%	56333	21225
libyal	c	112	16.50%	2463091	406482
libxlsxwriter	c	1	21.38%	20742	4435
libwebsockets	c	1	2.12%	34837	737
libunwind	c	1	43.62%	4411	1924
libucl	c	1			
libssh	c	8	43.48%	30600	13304
libspdm	c	69	52.50%	64655	33944
libsndfile	c	2	45.70%	25952	11860
libredwg	c	1	26.55%	62326	16547
libpg_query	c	1	11.68%	50714	5925
liboqs	c	2	37.89%	170335	64536
libmodbus	c	2	52.32%	151	79
liblouis	c	3	56.62%	10534	5964
libjpeg-turbo	c	20	60.73%	47686	28962
libiec61850	c	1	13.94%	3880	541
libhtp	c	1	55.92%	8955	5008
libfuse	c	1	66.54%	520	346
libdwarf	c	33	70.82%	45388	32145
libcue	c	1	60.20%	1779	1071
libconfig	c	1	67.07%	3234	2169
libcacard	c	3	75.45%	4510	3403
libbpf	c	1	22.95%	22684	5205
krb5	c	19	22.34%	61810	13806
kamailio	c	2	10.05%	85139	8558
jq	c	7	72.75%	40389	29381
jpegoptim	c	3	49.03%	1385	679
inih	c	1	91.85%	184	169
inchi	c	1			
igraph	c	26	62.25%	60062	37391
hwloc	c	1	15.12%	893	135
hiredis	c	1			
hdf5	c	2	17.63%	144627	25491
h3	c	22	94.23%	4579	4315
gss-ntlmssp	c	1	21.57%	6264	1351
gpsd	c	2	14.04%	37354	5245
gpac	c	4	16.37%	480284	78622
gdk-pixbuf	c	5	4.26%	227592	9689
gdbm	c	1	39.20%	11037	4327
fwupd	c	39	13.55%	226529	30691
fribidi	c	1	61.07%	1613	985
freerdp	c	7	15.65%	176765	27669
flex	c	2	27.78%	8944	2485
faad2	c	5	97.57%	12201	11905
edk2	c	12	25.72%	20518	5277
e2fsprogs	c	3			
dovecot	c	5	30.84%	45893	14155
dnsmasq	c	1	2.12%	25461	539
dbus-broker	c	1			
cyclonedds	c	6	29.54%	81032	23939
cups	c	9	30.78%	30109	9268
cryptsetup	c	3	10.15%	277407	28159
croaring	c	2	48.57%	17255	8381
cpuinfo	c	1	17.16%	2034	349
coturn	c	2	10.31%	2629	271
cmake	c	1	60.94%	7840	4778
clib	c	1	20.06%	4761	955
civetweb	c	3	19.21%	11450	2199
cgif	c	3	96.60%	1647	1591
cairo	c	5	18.66%	159138	29698
bluez	c	5	44.99%	14878	6693
bind9	c	11	21.48%	123801	26596
apache-httpd	c	7	14.65%	48933	7168
zstd	c++	21	77.93%	37978	29596
zopfli	c++	2	94.37%	2221	2096
znc	c++	1	1.24%	22498	279
zlib-ng	c++	7	55.98%	9711	5436
zlib	c++	11	80.63%	5291	4266
zeek	c++	33	49.19%	252350	124126
yoga	c++	1	46.69%	5453	2546
yara	c++	6	63.41%	30528	19359
yaml-cpp	c++	1	68.77%	3215	2211
yajl-ruby	c++	1			
xz	c++	4	73.68%	8620	6351
xvid	c++	1	33.47%	13782	4613
xpdf	c++	3	13.33%	126160	16823
xnnpack	c++	1	1.67%	98744	1647
xmlsec	c++	1	9.71%	112522	10926
xerces-c	c++	2	19.55%	96261	18818
wxwidgets	c++	1	5.70%	38261	2179
wuffs	c++	11	70.80%	44009	31157
wt	c++	5	4.24%	52991	2246
wpantund	c++	1	49.10%	18520	9094
wolfssl	c++	22	35.38%	945395	334518
woff2	c++	2	87.74%	3989	3500
wireshark	c++	7	59.35%	2828659	1678817
wget2	c++	20	13.82%	131356	18152
wget	c++	11	5.42%	143862	7793
wavpack	c++	1	84.45%	4181	3531
wasmedge	c++	2	27.80%	53575	14895
wabt	c++	5	68.42%	23078	15791
vulnerable-project	c++	1	92.11%	38	35
vorbis	c++	1	35.18%	8760	3082
valijson	c++	1	72.48%	6214	4504
uwebsockets	c++	12	85.43%	6484	5539
usrsctp	c++	3	37.92%	45813	17373
usbguard	c++	3	40.69%	3578	1456
uriparser	c++	6	80.92%	4727	3825
upx	c++	3	36.31%	29711	10789
unrar	c++	1			
unicorn	c++	13	45.04%	215203	96928
tremor	c++	1	69.36%	4820	3343
trafficserver	c++	8			
tpm2-tss	c++	242	0.38%	16995	65
tpm2	c++	1	22.84%	24716	5646
tor	c++	16	6.07%	397572	24142
tomlplusplus	c++	1	40.61%	6041	2453
tinyxml2	c++	2	39.91%	2721	1086
tinyusb	c++	3	0.00%	11380	0
tinyobjloader	c++	1	69.74%	1890	1318
tinygltf	c++	1	36.86%	14935	5505
tink	c++	1			
tidy-html5	c++	6	63.18%	20760	13117
tesseract-ocr	c++	2	19.43%	166032	32252
tensorflow	c++	60	2.86%	3923865	112347
tcmalloc	c++	1			
systemd	c++	46	29.37%	311068	91353
strongswan	c++	7	23.20%	55711	12923
stb	c++	2	81.79%	4608	3769
sqlite3	c++	1	79.35%	82061	65112
sql-parser	c++	1	73.85%	5645	4169
spotify-json	c++	1	70.36%	1019	717
spirv-tools	c++	7	46.28%	87481	40488
spirv-cross	c++	1	30.05%	5793	1741
spicy	c++	9	60.87%	26765	16292
spice-usbredir	c++	2	87.17%	1948	1698
speex	c++	4	82.80%	6663	5517
spdlog	c++	5	53.63%	5892	3160
solidity	c++	13	62.55%	133774	83682
snappy	c++	2	67.48%	1725	1164
sleuthkit	c++	11	35.95%	43497	15638
skia	c++	48	39.44%	609136	240264
skcms	c++	3	67.32%	3063	2062
simdutf	c++	4	59.34%	18577	11024
simdjson	c++	13	42.93%	11857	5090
simd	c++	1	1.87%	171225	3205
shaderc	c++	4	48.74%	160646	78298
serenity	c++	71	13.51%	467235	63133
sentencepiece	c++	1	2.09%	22802	477
s2opc	c++	4	7.79%	90484	7049
s2geometry	c++	1	10.47%	47144	4936
rocksdb	c++	2	25.08%	152501	38241
rnp	c++	8	37.91%	65900	24983
resiprocate	c++	2	27.41%	23448	6427
relic	c++	2			
readstat	c++	11	78.16%	11866	9274
re2	c++	1	31.91%	31578	10075
rdkit	c++	3			
rapidjson	c++	1	38.98%	2953	1151
radare2	c++	15			
qubes-os	c++	5			
quantlib	c++	3	1.31%	97459	1273
qt	c++	15	27.43%	465376	127648
qpid-proton	c++	2	45.92%	15726	7222
qpdf	c++	30	52.20%	43791	22861
pybind11	c++	1			
pugixml	c++	2	55.06%	7670	4223
proj4	c++	1	24.55%	153624	37721
powerdns	c++	7	20.42%	23975	4896
postgis	c++	3			
poppler	c++	16	17.68%	855263	151184
poco	c++	7	40.20%	59959	24105
pistache	c++	1	30.93%	10691	3307
pigweed	c++	15	32.89%	33833	11127
piex	c++	1	66.80%	2491	1664
picotls	c++	2	44.64%	7141	3188
php	c++	8	3.56%	401122	14293
phmap	c++	1	54.44%	1137	619
pffft	c++	2			
perfetto	c++	12	17.53%	250523	43925
pcre2	c++	9	82.14%	37698	30967
pcl	c++	1	42.53%	3066	1304
pcapplusplus	c++	5	46.81%	25690	12026
ots	c++	1	82.33%	20348	16753
osquery	c++	2	17.96%	43372	7788
opus	c++	24	83.76%	27633	23146
openweave	c++	6	10.29%	112616	11590
openvswitch	c++	6	25.29%	125337	31696
openthread	c++	5	39.98%	85237	34080
openssl	c++	139	43.07%	1361809	586477
openssh	c++	8	30.39%	35401	10758
opensc	c++	13	55.69%	96618	53803
opennavsurf-bag	c++	2	24.68%	6713	1657
openjpeg	c++	2			
openh264	c++	1	76.54%	19137	14647
openexr	c++	2	26.68%	57216	15264
opendnp3	c++	3	34.22%	24502	8384
opencv	c++	9	15.15%	358823	54361
opencensus-cpp	c++	5	8.95%	11489	1028
openbabel	c++	3	14.80%	84763	12545
open62541	c++	12	41.45%	84787	35143
onednn	c++	1	49.77%	213	106
ogre	c++	2	5.68%	106517	6045
oatpp	c++	1	41.75%	6395	2670
num-bigint	c++	1	16.24%	62523	10155
ntp	c++	2	7.30%	44680	3261
ntopng	c++	1	16.46%	67072	11039
nss	c++	16	35.42%	189615	67167
nodejs	c++	49	23.92%	89956	21513
njs	c++	1	20.11%	47535	9561
ninja	c++	1	39.97%	4624	1848
nghttp2	c++	3	53.65%	11897	6383
nettle	c++	2	53.55%	86478	46305
netcdf	c++	1			
net-snmp	c++	16	35.07%	58156	20397
nestegg	c++	1	81.00%	2095	1697
neomutt	c++	2			
ndpi	c++	56	74.74%	65578	49014
nccl	c++	1			
nanopb	c++	5	89.50%	2410	2157
mysql-server	c++	25	18.30%	22008	4027
myanmar-tools	c++	1	87.86%	173	152
mupdf	c++	1	10.36%	314517	32586
muparser	c++	1	51.77%	3027	1567
muduo	c++	1	14.80%	1912	283
msgpack-c	c++	1	28.92%	4422	1279
mruby	c++	2	34.69%	75758	26282
mpg123	c++	2	52.82%	9945	5253
mosh	c++	2	19.56%	3379	661
mongoose	c++	1	38.24%	5661	2165
monero	c++	13	10.73%	97039	10408
minizip	c++	2	66.09%	4188	2768
meshoptimizer	c++	1	67.50%	1280	864
mercurial	c++	10	26.09%	155518	40574
mbedtls	c++	20	31.22%	65404	20422
matio	c++	1			
mariadb	c++	1	3.39%	19701	668
mapserver	c++	3	7.32%	515424	37742
magic-enum	c++	1			
lzo	c++	3	91.92%	3849	3538
lzma	c++	9	86.98%	10598	9218
lz4	c++	10	77.65%	5444	4227
lwan	c++	6			
lodepng	c++	1	44.85%	4841	2171
llvm_libcxxabi	c++	1	40.60%	4158	1688
llvm_libcxx	c++	17	84.20%	443	373
llvm	c++	48	40.56%	6391967	2592546
lldb-eval	c++	2			
llamacpp	c++	8			
libzmq	c++	12	16.89%	28995	4897
libzip	c++	4	58.46%	8619	5039
libyaml	c++	9	85.84%	7301	6267
libxslt	c++	2	61.86%	63185	39089
libxml2	c++	11	70.14%	90993	63820
libxls	c++	1	73.39%	2601	1909
libxaac	c++	2	80.84%	136337	110212
libwebp	c++	12	47.72%	72641	34664
libvpx	c++	2	60.48%	32861	19875
libvnc	c++	1	19.32%	10834	2093
libvips	c++	17	52.92%	67888	35929
libusb	c++	1	1.06%	5658	60
libultrahdr	c++	3	34.20%	27157	9289
libtsm	c++	1			
libtpms	c++	1	23.22%	71946	16709
libtorrent	c++	13	22.01%	62490	13751
libtiff	c++	1	42.06%	50165	21098
libtheora	c++	1	75.67%	3716	2812
libteken	c++	1			
libtasn1	c++	9	59.40%	6779	4027
libstdcpp	c++	1	91.18%	102	93
libssh2	c++	1	29.78%	13032	3881
libsrtp	c++	1	60.65%	5593	3392
libspng	c++	3			
libspectre	c++	2			
libsoup	c++	5	7.61%	20992	1597
libsodium	c++	2	19.29%	8440	1628
libsass	c++	1	30.48%	20326	6195
libressl	c++	11	48.65%	177787	86491
libreoffice	c++	54	22.59%	3582235	809094
librdkafka	c++	1	57.68%	1269	732
librawspeed	c++	77	34.32%	72508	24882
libraw	c++	4	79.16%	37708	29848
libpsl	c++	12	19.23%	56116	10790
libprotobuf-mutator	c++	2	17.67%	182689	32279
libpng-proto	c++	4	25.57%	38527	9852
libpng	c++	1	43.31%	12834	5558
libplist	c++	4	38.34%	7377	2828
libphonenumber	c++	4	57.60%	10997	6334
libpcap	c++	3	55.82%	16596	9264
libmpeg2	c++	1	82.87%	10722	8885
libmicrohttpd2	c++	8	23.34%	22814	5324
libldac	c++	1	53.84%	2914	1569
libjxl	c++	11	48.38%	83267	40284
libigl	c++	1			
libidn2	c++	3	69.35%	7092	4918
libidn	c++	3	81.55%	2163	1764
libical	c++	2	41.93%	18528	7769
libhevc	c++	2	80.35%	145143	116629
libheif	c++	4	36.86%	26583	9799
libgit2	c++	8	23.69%	82188	19470
libgd	c++	11	33.68%	5365	1807
libfido2	c++	10			
libfdk-aac	c++	3	53.10%	80463	42725
libexif	c++	2	83.35%	5681	4735
libevent	c++	7	28.18%	19195	5410
libecc	c++	1	25.85%	171408	44310
libcups	c++	2	17.15%	17719	3039
libcoap	c++	4	7.88%	22698	1789
libcbor	c++	1	87.22%	2896	2526
libavif	c++	8	48.79%	338401	165107
libavc	c++	5	82.50%	135117	111477
libass	c++	1	22.08%	62075	13708
libarchive	c++	1	69.79%	37412	26108
libaom	c++	1	61.61%	81366	50130
leveldb	c++	1	75.61%	6684	5054
leptonica	c++	45	26.89%	194685	52346
lcms	c++	15	63.63%	20505	13047
lame	c++	1	39.95%	23780	9499
knot-dns	c++	31			
kmime	c++	1	12.66%	144080	18240
kimageformats	c++	24	21.98%	761949	167481
keystone	c++	27	46.98%	133346	62651
kcodecs	c++	1	4.06%	141199	5732
karchive	c++	9	8.18%	146348	11973
jwt-verify-lib	c++	1			
jsonnet	c++	3	29.20%	18586	5428
jsoncpp	c++	2	21.56%	8191	1766
jsoncons	c++	18	66.00%	21447	14156
json-c	c++	4			
json	c++	6	74.50%	8329	6205
jbig2dec	c++	1	93.54%	5668	5302
janus-gateway	c++	3	38.01%	6257	2378
jansson	c++	1	50.70%	5456	2766
janet	c++	1	9.19%	22565	2073
irssi	c++	4	14.34%	212357	30445
immer	c++	19	91.36%	11046	10092
imagemagick	c++	148	47.35%	584972	276995
icu	c++	33	47.45%	170305	80805
ibmswtpm2	c++	1	9.13%	20090	1834
hunspell	c++	2	75.29%	10056	7571
http-pattern-matcher	c++	1			
http-parser	c++	2	89.12%	1590	1417
htslib	c++	1	38.68%	50196	19415
hpn-ssh	c++	8	23.10%	36823	8507
hostap	c++	19	18.55%	135413	25120
hoextdown	c++	1	82.98%	4564	3787
highwayhash	c++	2	19.84%	1910	379
hermes	c++	1	15.61%	261566	40820
harfbuzz	c++	5	72.62%	60555	43973
haproxy	c++	2	1.62%	122823	1993
h2o	c++	4	16.08%	67605	10870
guetzli	c++	1	80.92%	5698	4611
gstreamer	c++	2	18.88%	251695	47508
grpc-httpjson-transcoding	c++	2	4.36%	103071	4490
grok	c++	1	21.77%	49980	10879
graphicsmagick	c++	125	29.16%	795949	232133
gnutls	c++	27	42.92%	114324	49067
gnupg	c++	4			
glslang	c++	1	43.14%	55291	23854
glog	c++	1	93.08%	867	807
glib	c++	27	11.70%	184344	21571
glaze	c++	10	49.69%	10926	5429
git	c++	11	10.68%	199902	21353
giflib	c++	3	29.39%	8281	2434
ghostscript	c++	17	43.03%	492292	211824
gfwx	c++	2	85.70%	1091	935
geos	c++	1	17.88%	55204	9871
gdal	c++	50	39.31%	1039620	408719
fuzztest-raksha	c++	1			
frr	c++	4	3.64%	349605	12716
freetype2	c++	33	65.00%	105386	68499
freeradius	c++	9	29.20%	41843	12218
freeimage	c++	1	10.34%	196049	20281
fmt	c++	6	81.33%	5823	4736
fluent-bit	c++	29	12.40%	434855	53906
flatbuffers	c++	7	51.29%	14505	7440
flac	c++	9	87.87%	34613	30416
firestore	c++	4	10.44%	24056	2511
fio	c++	1			
file	c++	3	78.55%	10836	8512
fftw3	c++	1	16.15%	76802	12401
ffms2	c++	1			
ffmpeg	c++	1109	61.87%	772930	478174
fast_float	c++	1			
fast-dds	c++	2	0.62%	242568	1499
exprtk	c++	1	41.91%	23503	9849
expat	c++	13	69.18%	12023	8317
exiv2	c++	2	66.66%	33696	22461
espeak-ng	c++	2			
esp-v2	c++	1			
envoy	c++	66	20.47%	1678095	343527
elfutils	c++	3	25.59%	34010	8704
eigen	c++	2	16.98%	5700	968
ecc-diff-fuzzer	c++	12			
easywsclient	c++	1	24.03%	362	87
duckdb	c++	1	24.73%	382833	94692
dropbear	c++	15	55.68%	27035	15053
draco	c++	4	71.22%	8800	6267
double-conversion	c++	1	55.03%	1928	1061
dng_sdk	c++	5	68.83%	37105	25539
dav1d	c++	2	79.32%	15937	12642
cxxopts	c++	1	51.58%	1266	653
curl	c++	18	25.47%	338629	86245
cups-filters	c++	1			
cryptofuzz	c++	5	30.92%	702405	217187
crow	c++	3	8.64%	8055	696
cpython3	c++	11	39.10%	264283	103347
cppitertools	c++	1	96.69%	423	409
cppcheck	c++	1			
cpp-httplib	c++	1	32.05%	6553	2100
connectedhomeip	c++	7	10.29%	201883	20773
cmark	c++	1	95.18%	14707	13998
clamav	c++	3			
cjson	c++	1	44.00%	2325	1023
circl	c++	2			
cfengine	c++	2			
cel-cpp	c++	1			
cctz	c++	1	84.05%	2388	2007
casync	c++	1	10.01%	1678	168
capstone	c++	2	88.14%	497571	438536
capnproto	c++	1	6.47%	37397	2420
c-blosc2	c++	4	28.65%	47827	13703
c-blosc	c++	2			
c-ares	c++	2	39.33%	18000	7080
bzip2	c++	4	92.97%	2661	2474
brunsli	c++	2	89.34%	7043	6292
brpc	c++	10	8.19%	148782	12183
brotli	c++	1	79.79%	3311	2642
botan	c++	34	41.25%	79183	32660
boringssl	c++	37	37.22%	114372	42565
boost-json	c++	4	44.66%	20115	8983
boost-beast	c++	3	48.45%	23357	11316
boost	c++	14	57.57%	24905	14339
bls-signatures	c++	2			
bloaty	c++	1	15.93%	204590	32595
bitcoin-core	c++	227	39.19%	214099	83897
binutils	c++	26	32.50%	621526	202026
bignum-fuzzer	c++	5	11.36%	208845	23730
behaviortreecpp	c++	3	48.88%	15500	7576
bearssl	c++	1	35.11%	74871	26284
avahi	c++	6	76.66%	3046	2335
augeas	c++	3			
astc-encoder	c++	1	49.76%	2968	1477
assimp	c++	1	12.71%	124667	15840
aspell	c++	1	67.96%	14390	9780
arrow	c++	4	28.70%	147645	42372
args	c++	1	22.66%	1796	407
arduinojson	c++	2			
apache-logging-log4cxx	c++	16	39.52%	15484	6119
ampproject	c++	1	22.65%	41125	9314
alembic	c++	1	12.10%	12569	1521
ada-url	c++	6	75.68%	8828	6681
abseil-cpp	c++	1			
ygot	go	1	0.50%	66296	333
volcano	go	2	54.04%	1188	642
vitess	go	27	10.57%	257292	27199
uint256	go	2	96.58%	1668	1611
u-root	go	7	61.82%	1032	638
timestamp-authority	go	2	0.00%	127	0
time	go	7	75.81%	2071	1570
tidb	go	3	23.42%	28996	6792
tendermint	go	3	0.00%	0	0
teleport	go	36	14.88%	131890	19621
tailscale	go	1	94.87%	156	148
syzkaller	go	4	40.53%	17200	6972
smt	go	2	59.62%	421	251
skipper	go	9	15.74%	15929	2507
sigstore-go	go	9	40.00%	425	170
sigstore	go	10	0.00%	0	0
scorecard-web	go	2	22.17%	397	88
runc	go	10	24.90%	8318	2071
roughtime	go	2	82.99%	488	405
rekor	go	38	43.73%	2456	1074
radon	go	1	90.24%	41	37
quic-go	go	6	23.35%	9583	2238
pulumi	go	3	16.12%	4224	681
publicsuffix-list	go	1	0.00%	150	0
proton-bridge	go	5	69.05%	433	299
protocompile	go	1	45.28%	8801	3985
protoc-gen-validate	go	1	1.10%	1086	12
prometheus	go	4	47.45%	27760	13172
pborman-uuid	go	1	80.49%	164	132
p9	go	1	49.47%	2351	1163
ossf-scorecard	go	1	3.20%	8851	283
openyurt	go	2	35.95%	2125	764
opentelemetry-go-contrib	go	2	0.00%	0	0
opentelemetry-go	go	1	98.13%	481	472
opentelemetry	go	39	80.55%	6138	4944
openkruise	go	19	66.81%	3378	2257
opencensus-go	go	1	100.00%	38	38
notary	go	16	36.42%	4731	1723
ngolo-fuzzing-x	go	72	19.06%	78600	14980
ngolo-fuzzing	go	111	35.49%	160243	56876
nats	go	2	97.22%	790	768
mxj	go	1	72.69%	2025	1472
multierr	go	1	100.00%	100	100
mtail	go	1	0.00%	0	0
mongo-go-driver	go	1	75.53%	5226	3947
moby	go	12	5.56%	35813	1991
minify	go	18	76.50%	9381	7176
metallb	go	3	58.83%	1234	726
lotus	go	5	23.20%	3271	759
loki	go	1	0.00%	0	0
litmuschaos	go	24	18.20%	676	123
linkerd2	go	12	30.84%	12461	3843
lima	go	8	37.72%	1246	470
kyverno	go	15	43.89%	4623	2029
kubevirt	go	14	52.44%	8802	4616
kubernetes-cluster-api	go	18	89.49%	1693	1515
kubernetes	go	47	13.07%	105592	13801
kubeedge	go	9	23.84%	5554	1324
kubearmor	go	2	1.14%	3768	43
knative	go	22	43.16%	9279	4005
juju	go	1	3.43%	8904	305
jsonparser	go	14	96.25%	773	744
json-patch	go	2	88.66%	582	516
istio	go	55	33.54%	52080	17470
ipfs	go	3	86.77%	189	164
hugo	go	1	24.53%	2943	722
helm	go	37	47.61%	10472	4986
hcl	go	6	0.00%	0	0
grpc-go	go	2	32.00%	125	40
grpc-gateway	go	1	65.00%	300	195
gosnmp	go	1	46.58%	2413	1124
gopsutil	go	1	24.63%	881	217
gopacket	go	1	80.75%	8906	7192
gonids	go	1	96.04%	1035	994
golang-protobuf	go	3	45.98%	17825	8196
golang-appengine	go	1	52.41%	166	87
golang	go	82	60.01%	15012	9008
gogo-protobuf	go	1	0.00%	0	0
gobgp	go	7	57.76%	9367	5410
go-yaml	go	1	75.82%	4247	3220
go-toml	go	1	72.59%	2474	1796
go-sqlite3	go	1	16.21%	1363	221
go-shlex	go	1	72.03%	143	103
go-sftp	go	1	0.00%	0	0
go-redis	go	1	2.17%	10358	225
go-readline	go	1	16.93%	1896	321
go-pprof	go	1	70.41%	1940	1366
go-ole	go	1	19.62%	525	103
go-ldap	go	3	53.92%	2205	1189
go-json-iterator	go	1	45.80%	3967	1817
go-humanize	go	1	99.69%	321	320
go-git	go	5	73.15%	4399	3218
go-dns	go	2	79.56%	9672	7695
go-dhcp	go	3	89.02%	2003	1783
go-coredns	go	7	47.07%	5810	2735
go-containerregistry	go	1	75.23%	222	167
go-coap	go	4	67.67%	767	519
go-cmp	go	1	0.00%	0	0
go-attestation	go	3	14.05%	1665	234
gluon	go	4	69.90%	1721	1203
gitea	go	2	0.00%	0	0
gcloud-go	go	1	0.00%	0	0
gateway	go	1	0.00%	0	0
fsnotify	go	1	66.47%	340	226
fluxcd	go	34	71.65%	2028	1453
fastjson	go	1	0.00%	0	0
fasthttp	go	4	15.82%	7079	1120
fabric	go	7	14.84%	4172	619
expr	go	1	92.31%	13	12
etcd	go	14	0.00%	0	0
dragonfly	go	2	29.72%	3806	1131
distribution	go	12	48.22%	3824	1844
dgraph	go	1	80.64%	2428	1958
demangle	go	1	42.68%	4735	2021
dapr	go	44	0.00%	0	0
cubefs	go	7	24.73%	61371	15177
crossplane	go	10	67.81%	2277	1544
cri-o	go	6	39.33%	3140	1235
cosmos-sdk	go	12	40.22%	4552	1831
cosign	go	6	44.62%	1757	784
containerd	go	28	59.32%	3525	2091
config-validator	go	1	46.66%	1301	607
compress	go	15	64.03%	21474	13750
cockroachdb	go	6	54.78%	12772	6997
clock	go	1	87.71%	179	157
cilium	go	13	5.31%	72458	3845
cert-manager	go	11	23.22%	7620	1769
cel-go	go	1	55.24%	16958	9367
cascadia	go	1	50.76%	920	467
caddy	go	7	24.95%	14229	3550
burntsushi-toml	go	1	63.27%	1764	1116
blackfriday	go	1	92.79%	2399	2226
atomic	go	1	100.00%	243	243
astro-compiler	go	2	56.46%	441	249
zxing	java	2	48.79%	21062	10277
zt-zip	java	1	11.55%	3801	439
zookeeper	java	4	1.92%	75439	1450
zip4j	java	1	16.08%	4516	726
yamlbeans	java	2			
xz-java	java	1	36.15%	4553	1646
xstream	java	1	11.03%	32554	3592
xnio-api	java	1	0.28%	38438	106
xmlunit	java	1	2.94%	18469	543
xmlpull	java	1	5.98%	1138	68
xerces	java	1	7.63%	71926	5486
woodstox	java	1	12.66%	34596	4379
univocity-parsers	java	2	9.82%	18912	1858
unirest-java	java	2			
undertow	java	1	0.81%	87360	709
tyrus	java	1	0.89%	27188	242
twitter4j	java	1	5.76%	7533	434
twelve-monkeys	java	1	0.13%	54114	69
threetenbp	java	1	9.65%	30876	2980
tablesaw	java	1	4.89%	31175	1524
swagger-core	java	1			
struts	java	1			
stringtemplate4	java	1	16.10%	8303	1337
sqlite-jdbc	java	1	11.52%	11007	1268
spring-webflow	java	1	1.30%	122751	1596
spring-retry	java	1	4.53%	2031	92
spring-ldap	java	1	21.96%	1175	258
spring-integration	java	1	19.74%	4407	870
spring-data-redis	java	3	2.42%	403821	9773
spring-data-mongodb	java	1	1.43%	380473	5442
spring-data-keyvalue	java	1	8.80%	2875	253
spring-data-jpa	java	1			
spring-cloud-stream	java	1	0.47%	28339	134
spring-cloud-sleuth-brave	java	1	0.04%	1815587	721
spring-cloud-netflix	java	1	0.68%	4582	31
spring-cloud-config	java	2	100.00%	58	58
spring-cloud-commons	java	1	0.16%	14598	24
spring-amqp	java	1	0.00%	0	0
spatial4j	java	1	12.73%	6063	772
snappy-java	java	2	24.16%	1676	405
snakeyaml	java	2	46.98%	6405	3009
slf4j-api	java	1	2.27%	7696	175
sketches-core	java	2	0.36%	64176	234
sigstore-java	java	14	8.11%	28028	2272
servo-core	java	1	3.07%	6488	199
rxjava	java	1	0.13%	47734	62
rome	java	1	19.63%	18085	3550
roaring-bitmap	java	1	4.49%	19197	862
rhino	java	1	9.48%	52831	5010
retrofit	java	2	1.80%	136628	2456
reload4j	java	1	2.26%	15017	340
reflections	java	2	5.89%	2341	138
rdf4j	java	1	2.34%	307716	7207
quartz	java	1	2.63%	16554	436
qdox	java	1	20.47%	16537	3385
protobuf-java	java	1	0.00%	8	0
presto	java	1	0.73%	608229	4457
powsybl-java	java	6	2.06%	344867	7090
plexus-utils	java	2	0.14%	8296	12
pdfbox	java	2	9.92%	79702	7904
osgi	java	3	0.41%	165226	674
opencsv	java	1			
opencensus-java	java	1	5.73%	7638	438
open-json	java	1	2.82%	3010	85
ohc	java	1			
nimbus-jwt	java	1	12.00%	13777	1653
netty	java	4	98.11%	106	104
mybatis-3	java	2	0.74%	34858	258
mvel	java	1	29.78%	26038	7754
micronaut	java	38			
metadata-extractor	java	1	48.13%	19401	9337
maven-model	java	1	4.94%	134860	6657
maven	java	2	32.11%	1859	597
mariadb-connector-j	java	1	0.00%	47619	0
lucene	java	3	24.10%	128392	30943
logback	java	1	22.44%	16383	3677
log4j2	java	5			
kryo	java	3	7.10%	18077	1284
kie-soup	java	1	2.43%	17057	414
keycloak	java	24	2.89%	135027	3907
junrar	java	1	0.56%	5731	32
jul-to-slf4j	java	1	0.18%	7697	14
jts	java	1	7.04%	73875	5200
jsqlparser	java	1	11.73%	57543	6749
jsoup	java	2	48.71%	9872	4809
jsonpath	java	1	44.11%	4806	2120
jsonp-api	java	2	0.98%	16095	157
json2avro	java	1			
json-smart-v2	java	1	10.17%	5990	609
json-simple	java	2	24.04%	1589	382
json-sanitizer	java	3	48.41%	1198	580
json-java	java	1	9.76%	2868	280
json-flattener	java	2	32.73%	1604	525
jsign	java	4	6.06%	11686	708
jsemver	java	1	10.12%	2124	215
jsch	java	1			
jose4j	java	1	2.35%	15403	362
jopt-simple	java	1	8.43%	4474	377
jooq	java	1			
joni	java	2	42.80%	10884	4658
jolt	java	1	2.14%	3174	68
joda-time	java	1	0.88%	59485	522
joda-convert	java	3	18.52%	1685	312
jmh	java	1	0.13%	171393	228
jline3	java	1	22.25%	28967	6445
jimfs	java	1	10.15%	10339	1049
jfreechart	java	3	7.38%	51456	3795
jflex	java	1	38.74%	10295	3988
jettison	java	1	14.55%	2302	335
jedis	java	1	0.71%	75664	539
jdom	java	1	3.41%	30119	1026
jboss-logging	java	1	6.31%	2093	132
jaxb	java	1	0.61%	66004	404
javassist	java	1			
javapoet	java	1	11.86%	5588	663
javaparser	java	1	27.69%	47507	13154
javacpp	java	1	3.31%	17173	569
java-xmlbuilder	java	1	22.53%	861	194
java-uuid-generator	java	1	10.40%	3201	333
java-jwt	java	1	0.00%	0	0
java-diff-utils	java	1	5.46%	3317	181
jansi	java	1	26.01%	2211	575
janino	java	1	13.26%	21685	2875
jakarta-mail-api	java	1	1.55%	7404	115
jackson-modules-java8	java	1	3.67%	10921	401
jackson-datatypes-collections	java	7	62.21%	5282	3286
jackson-datatype-joda	java	3	67.41%	1083	730
jackson-dataformats-text	java	6	52.81%	8013	4232
jackson-dataformats-binary	java	13	28.90%	17092	4939
jackson-dataformat-xml	java	5	43.44%	3400	1477
jackson-databind	java	7	17.00%	205174	34875
jackson-core	java	5	15.30%	55241	8452
itext7	java	1	4.53%	67677	3066
ion-java	java	2	0.00%	0	0
httpcomponents-core	java	1	35.92%	6144	2207
httpcomponents-client	java	6	0.30%	99189	301
htmlunit	java	1			
hsqldb	java	3	25.26%	95141	24032
hive	java	1			
hikaricp	java	4	2.05%	6960	143
hibernate-validator	java	1	2.30%	48745	1120
hdrhistogram	java	1	3.65%	7013	256
hamcrest	java	1	42.14%	1680	708
hadoop	java	2	0.02%	1413142	340
h2database	java	6	16.40%	82875	13592
gwt	java	2	1.41%	210633	2962
guice	java	1	5.42%	39563	2143
guava	java	7	1.92%	136389	2612
gson	java	3	6.28%	18325	1151
greenmail	java	1	5.20%	6037	314
graphql-java	java	1	3.87%	89956	3478
g-oauth-java-client	java	1	1.45%	2887	42
g-http-java-client	java	1	3.79%	14365	544
g-auth-library-java	java	2	4.58%	21008	963
fuzzywuzzy	java	2	41.56%	912	379
flyway	java	1	0.47%	21093	100
feign	java	2			
fastjson2	java	1	4.44%	197092	8752
fastcsv	java	1	19.96%	1578	315
exp4j	java	1	11.73%	3401	399
evo-inflector	java	1	40.41%	344	139
dropwizard	java	1			
dom4j	java	1	5.91%	9502	562
docker-client	java	1	0.10%	16997	17
curvesapi	java	1	11.30%	2876	325
cron-utils	java	1	10.42%	8396	875
checkstyle	java	1	13.76%	39552	5441
cglib	java	1	26.26%	8879	2332
cbor-java	java	1	49.36%	1163	574
calcite-avatica	java	2	0.02%	42061	10
calcite	java	2	7.01%	259348	18193
caffeine	java	1			
brotli-java	java	1	62.77%	1891	1187
bc-java	java	4	41.93%	28825	12087
avro	java	1	0.00%	0	0
async-http-client	java	1	11.32%	15020	1700
aspectj	java	1	2.30%	148217	3403
arrow-java	java	2	0.61%	127302	773
args4j	java	1	5.93%	2478	147
apache-poi	java	16	9.29%	491783	45671
apache-felix-dev	java	1	2.84%	5837	166
apache-cxf	java	4	0.46%	403433	1836
apache-commons-validator	java	4	6.25%	9018	564
apache-commons-text	java	2	6.00%	18214	1092
apache-commons-net	java	5	2.90%	14929	433
apache-commons-logging	java	1	22.01%	22050	4853
apache-commons-lang	java	14	8.83%	64090	5657
apache-commons-jxpath	java	1	50.67%	12117	6140
apache-commons-io	java	10	34.09%	9508	3241
apache-commons-imaging	java	5	26.49%	16941	4488
apache-commons-geometry	java	4	44.65%	4840	2161
apache-commons-csv	java	1	7.73%	6207	480
apache-commons-configuration	java	8	13.22%	10844	1434
apache-commons-compress	java	16	41.87%	25474	10665
apache-commons-collections	java	1	0.22%	40503	91
apache-commons-codec	java	9	61.22%	4868	2980
apache-commons-cli	java	1	17.22%	1975	340
apache-commons-beanutils	java	1	0.89%	15015	133
apache-commons-bcel	java	1			
apache-axis2	java	1	2.90%	364217	10566
antlr4-java	java	1	14.30%	46283	6620
antlr3-java	java	1	17.84%	59601	10632
angus-mail	java	2			
tint	N/A	8	26.17%	303832	79502
skia-ftz	N/A	37	36.18%	582332	210706
scrapy	N/A	1	17.09%	3311	566
rinja	N/A	4	73.13%	3967	2901
protoreflect	N/A	1	14.29%	6038	863
gvisor	N/A	1	0.00%	0	0
commons-validator	N/A	4	2.07%	28075	580
commons-net	N/A	5	7.18%	7253	521
zipp	python	1	58.50%	294	172
yarl	python	1	61.37%	1662	1020
xmltodict	python	1	67.76%	335	227
xlsxwriter	python	1	28.30%	11390	3223
xlrd	python	1	45.34%	4910	2226
wtforms	python	1	51.29%	2798	1435
wheel	python	1	3.12%	32	1
websockets	python	3	62.45%	831	519
websocket-client	python	2	32.11%	1504	483
w3lib	python	3	53.11%	322	171
validators	python	1	61.75%	813	502
urllib3	python	2	32.87%	3855	1267
urlextract	python	1	51.16%	2191	1121
uritemplate	python	1	68.22%	387	264
underscore	python	1	84.78%	1045	886
unblob	python	1	49.54%	36018	17845
ujson	python	3	41.39%	16236	6720
typing_extensions	python	1	29.11%	1824	531
tqdm	python	1	21.02%	2849	599
toolz	python	1	60.62%	1290	782
toolbelt	python	1	27.33%	9948	2719
tomlkit	python	2	75.53%	2967	2241
tomli	python	1	66.80%	729	487
toml	python	2	20.31%	699	142
tinycss2	python	2	73.49%	1011	743
textdistance	python	1	63.53%	1434	911
tensorflow-addons	python	1	28.32%	349944	99098
stack_data	python	1	31.66%	2372	751
sqlparse	python	2	38.36%	1671	641
sqlalchemy_jsonfield	python	1	40.99%	53307	21852
sqlalchemy-utils	python	3	42.97%	75494	32436
sqlalchemy	python	1	45.37%	17808	8080
soupsieve	python	4	41.22%	7885	3250
smart_open	python	4	19.84%	3786	751
six	python	1	52.98%	553	293
simplejson	python	2	27.08%	912	247
sigstore-python	python	1	39.32%	28985	11397
setuptools	python	1	10.64%	1334	142
scipy	python	2	22.82%	30942	7062
scikit-learn	python	1	22.39%	87290	19545
scapy	python	1	49.86%	38270	19080
sacremoses	python	4	23.79%	12423	2956
rich	python	1	66.42%	19701	13086
rfc3967	python	1	72.77%	951	692
retry	python	1	38.76%	387	150
requests	python	1	42.21%	7917	3342
redis-py	python	5	25.20%	10372	2614
pyzmq	python	1	65.77%	1674	1101
pyyaml	python	3	77.38%	3732	2888
pyxdg	python	4	25.12%	1206	303
pyvex	python	1	35.63%	20068	7150
pytz	python	1	12.57%	2236	281
python3-openid	python	1	48.48%	198	96
python-tabulate	python	1	65.14%	958	624
python-rsa	python	1	55.50%	737	409
python-rison	python	2			
python-pypdf	python	1	34.02%	9779	3327
python-prompt-toolkit	python	2	29.51%	13343	3938
python-phonenumbers	python	1	77.44%	3652	2828
python-pathspec	python	1	56.31%	634	357
python-nvd3	python	1	28.30%	6259	1771
python-nameparser	python	1			
python-multipart	python	3	43.03%	918	395
python-markdownify	python	1	43.91%	7781	3417
python-markdown	python	1			
python-lz4	python	1	29.90%	291	87
python-jose	python	2	32.33%	9632	3114
python-hyperlink	python	2	62.38%	1313	819
python-graphviz	python	1	67.50%	1154	779
python-future	python	4	32.16%	8710	2801
python-fastjsonschema	python	1	19.60%	791	155
python-email-validator	python	1	82.74%	1037	858
python-ecdsa	python	3	56.81%	3438	1953
pytest-py	python	1	12.47%	2302	287
pytables	python	2	24.00%	32058	7694
pyrsistent	python	1	42.00%	1962	824
pyparsing	python	1	45.56%	4137	1885
pyodbc	python	2	54.17%	48	26
pynacl	python	2	24.14%	3430	828
pymysql	python	1	51.78%	2584	1338
pyjwt	python	1	44.25%	1182	523
pyjson5	python	1	75.32%	948	714
pygments	python	2	72.05%	12946	9328
pydateutil	python	3	39.66%	19235	7628
pycrypto	python	4	58.21%	1761	1025
pycparser	python	1	39.88%	4506	1797
pyasn1-modules	python	1	69.49%	5903	4102
pyasn1	python	1			
py-serde	python	1	49.04%	728	357
psycopg2	python	1	46.30%	663	307
psutil	python	1	30.25%	2724	824
psqlparse	python	1	52.76%	961	507
protobuf-python	python	1	23.03%	5654	1302
proto-plus-python	python	1	27.94%	7923	2214
ply	python	1	63.79%	1980	1263
pip	python	1	33.74%	33603	11339
pillow	python	2	49.16%	14187	6974
pikepdf	python	1	34.12%	6082	2075
pendulum	python	1	40.07%	7329	2937
pem	python	1	51.03%	194	99
pdoc	python	1	40.04%	9386	3758
pdfplumber	python	1			
pdfminersix	python	3	67.90%	8432	5725
pathlib2	python	3	35.02%	871	305
pasta	python	1	71.02%	2119	1505
parso	python	4	77.47%	4465	3459
parsimonious	python	1	48.15%	3655	1760
parse	python	1	76.92%	572	440
paramiko	python	1	27.76%	18983	5270
pandas	python	9	26.95%	195744	52755
packaging	python	1	11.02%	5011	552
oscrypto	python	2	51.46%	7701	3963
orjson	python	1	52.94%	34	18
oracle-py-cx	python	1	69.23%	78	54
opt_einsum	python	1	20.36%	609	124
openpyxl	python	8	80.15%	15215	12195
opencensus-python	python	1	35.69%	8604	3071
openapi-schema-validator	python	2	33.54%	9658	3239
olefile	python	1	43.75%	1184	518
oauthlib	python	1	45.84%	4385	2010
oauth2	python	1	53.25%	646	344
numpy	python	6	20.06%	638	128
numexpr	python	1	20.39%	618	126
ntlm2	python	2	36.10%	13268	4790
ntlm-auth	python	1	45.48%	1029	468
nfstream	python	1	19.52%	1788	349
networkx	python	3	19.31%	43335	8367
netaddr-py	python	1	29.13%	2513	732
nbformat	python	1	26.68%	14679	3917
nbclassic	python	1	28.48%	39064	11126
mutagen	python	1	61.46%	9416	5787
multidict	python	1			
msgpack-python	python	1	57.04%	142	81
msal	python	2	27.19%	9593	2608
mrab-regex	python	3			
more-itertools	python	1	22.52%	2118	477
model-transparency	python	3	36.05%	56579	20399
mdurl	python	1	83.83%	334	280
mdit-py-plugins	python	2	78.14%	5577	4358
mccabe	python	1	63.81%	268	171
markupsafe	python	1			
markdown-it-py	python	2	78.79%	3668	2890
mako	python	1	52.45%	2143	1124
lxml	python	6	45.83%	336	154
"""

if __name__ == "__main__":
    # データを使って関数を呼び出す
    filtered_names = extract_c_cpp_project_names(data)
    
    # 結果を出力
    # print("--- 抽出されたCおよびC++プロジェクト名 ---")
    # for name in filtered_names:
    #     print(name)
    
    # print(f"\n見つかったC/C++プロジェクトの合計: {len(filtered_names)}")
    with open("c_cpp_projects.txt", "w") as f:
        f.write("\n".join(filtered_names))
    print(f"抽出されたCおよびC++プロジェクト名は 'c_cpp_projects.txt' に保存されました。")
    print(f"見つかったC/C++プロジェクトの合計: {len(filtered_names)}")
