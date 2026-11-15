from PIL import Image
import matplotlib.pyplot as plt


image_ruta='imagen_1.png'

try:
  imagen_original=Image.open(image_ruta).convert('L')  
except FileNotFoundError:
  print(f"La imagen no se encuentró en la ruta: {image_ruta}")
  exit(1)
  

ancho, alto=imagen_original.size

######################################################################################################
##### Histograma

histograma=[0]*256 
for y in range(alto):
  for x in range(ancho):
    pixel=imagen_original.getpixel((x,y))
    histograma[pixel]+=1 #Numero de veces
    


########################################################################################################
# Calcular probailidad
suma_prob = []
total=ancho*alto
probabilidad= [contador/total for contador in histograma]

acumulado = 0
for prob in probabilidad:
    acumulado += prob
    suma_prob.append(acumulado)


#########################################################################################################
# Transformación
transforma=[0]*256
for i in range(256):
  transforma[i]=int(255*suma_prob[i])

ecualizada=Image.new('L', (ancho,alto))

for y in range(alto):
  for x in range(ancho):
    pixel=imagen_original.getpixel((x,y))
    nuevo_pixel=transforma[pixel]
    ecualizada.putpixel((x,y),nuevo_pixel)
    
    
    
plt.figure(figsize=(8,5))
plt.subplot(1,2,1)
plt.imshow(imagen_original, cmap='gray')
plt.axis('off')
plt.title('Imagen Original')
plt.subplot(1,2,2)
plt.imshow(ecualizada, cmap='gray') 
plt.title('Imagen Ecualizada')
plt.axis('off')


######################################################################################################
#### Histograma ecualizado
histograma_ecualizado=[0]*256 
for y in range(alto):
  for x in range(ancho):
    pixel=ecualizada.getpixel((x,y))
    histograma_ecualizado[pixel]+=1 
    
plt.figure(figsize=(8,5))
plt.subplot(1,2,1)
plt.bar(range(256),histograma)
plt.title('Histograma normal')
plt.subplot(1,2,2)
plt.bar(range(256),histograma_ecualizado)
plt.title('Histograma Ecualizado')





ecualizada.save("Reto_ecualizada.jpg", "JPEG", quality=75)