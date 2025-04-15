# TEKNOFEST'İN SUNUCUSUNUN TEST İÇİN OLAN ÖRNEĞİ

Sunucuyu kurmak için **UBUNTU 24.04** kullandığım için kurulumu ona göre anlatıcam.

```
sudo apt update

sudo apt install python3-pip

sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate
```

Venv'i aktif ettikten sonra komut istemi şu şekilde olmalı: 
```(venv) root@ubuntu:```

```
pip install -r requirements.txt

python app.py
```