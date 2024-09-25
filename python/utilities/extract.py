from PIL import Image
import os

path = "./myData"

i=0
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if file.endswith('.png'):
            pat=os.path.join(r, file)
            with Image.open(pat) as im:
                if im.size!=(100, 100):
                    im=im.resize((100, 100),Image.LANCZOS)
                im.save(pat.replace(".png",".jpg"))
            os.remove(pat)
            i+=1
            print(i,end='\r')
        elif file.endswith('.jpg'):
            pat=os.path.join(r, file)
            with Image.open(pat) as im:
                if im.size!=(100, 100):
                    im=im.resize((100, 100),Image.LANCZOS)
                    im.save(pat)
                    i+=1
                    print(i,end='\r')

