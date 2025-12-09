#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################


import numpy as np, os
import sys
import holopy as hp
import matplotlib.pylab as plt
from PIL import Image, ImageSequence
from holopy.core.process import center_find
import tifffile
import matplotlib as mpl
from pathlib import Path

mpl.rcParams['figure.dpi'] = 300


###############################################################################
def ProgressBar(percent,barLen=50):
    sys.stdout.write('\r')
    progress = ""
    for i in range(barLen):
        if i<= int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.write("\r")
    sys.stdout.flush()



def check_and_unwrap(sequence_path):
    dir_content = os.listdir(sequence_path)  # Getting the list of directories

    if len(dir_content) == 0:  # Checking if the list is empty or not
        print('Empty directory, unwrapping image sequence.')  # If the directory is empty, unwrap the image sequence

        if not sequence_path.endswith('background'):
            with Image.open(os.path.join(sequence_path, 'sequence/image_verysmall.tif')) as im:
                index = 1
                sequence_length = sum(1 for _ in ImageSequence.Iterator(im))

                for frame in ImageSequence.Iterator(im):
                    ProgressBar(1 - (sequence_length - index) / sequence_length)  # Progress bar visualized in command line
                    frame.save(os.path.join(sequence_path, f'image/image_{index:04d}.tif'))
                    index += 1
                print()

        else:  # If the sequence path ends with 'background'
            with Image.open(os.path.join(sequence_path, 'sequence/image.tif')) as im:
                index = 1
                sequence_length = sum(1 for _ in ImageSequence.Iterator(im))

                for frame in ImageSequence.Iterator(im):
                    ProgressBar(1 - (sequence_length - index) / sequence_length)  # Progress bar visualized in command line
                    frame.save(os.path.join(sequence_path, f'image/image_{index:04d}.tif'))
                    index += 1
                print()

    else:
        print('Not empty directory, images already unwrapped.\n')  # If the directory is not empty, continue
        


# - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
def extract_tiff_metadata(file_path):
    """
    Estrae e stampa i metadati (tag) di ogni frame in un TIFF multipagina.
    """
    try:
        # 1. Apre il file TIFF
        with tifffile.TiffFile(file_path) as tif:
            num_pages = len(tif.pages)
            print(f"File: {os.path.basename(file_path)}")
            print(f"Identificate {num_pages} pagine (immagini).")

            # 2. Itera su ogni pagina
            for i, page in enumerate(tif.pages):
                
                print(f"\n--- Frame/Pagina {i+1} ---")
                
                # Accedi ai tag (metadati)
                # I tag sono spesso un dizionario con chiavi numeriche/stringa
                # I metadati specifici (es. timestamp, esposizione) sono
                # spesso inclusi nel tag ImageDescription (codice 270) in formato JSON o stringa.
                
                print(f"Dimensioni del Frame: {page.shape}")
                
                # Stampa alcuni tag standard
                print(f"Tipo di dati (dtype): {page.dtype}")
                
                # 3. Estrazione dei metadati specifici (ImageDescription, se presenti)
                try:
                    desc_tag = page.tags["ImageDescription"].value
                    print(f"ImageDescription (Metadati): {desc_tag[:100]}...")
                    # Se il desc_tag è JSON, puoi usare json.loads(desc_tag) per analizzarlo
                except KeyError:
                    print("Tag 'ImageDescription' (Metadati) non trovato.")
                    
                # Stampa tutti i tag (opzionale)
                for tag in page.tags:
                    print(f"  {tag.name}: {tag.value}")

    except FileNotFoundError:
        print(f"Errore: File non trovato a {file_path}")
    except Exception as e:
        print(f"Si è verificato un errore durante l'elaborazione del file: {e}")
    
    
def check_and_unwrap_new(input_file_path, save_label=False, output_dir = None):
    """
    Analizza un file TIFF, carica i dati in memoria e opzionalmente salva i frame su disco.
    
    Args:
        input_file_path (str): Path del file TIFF.
        save_label (bool): Se True, salva i singoli frame nella cartella 'unwrapped/'.
        
    Return:
        is_multipage (bool): True se è uno stack, False se è una immagine singola.
        image_data_list (list): Una lista di numpy.array (le matrici delle immagini).
    """
    
    input_path = Path(input_file_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File non trovato: {input_file_path}")

    image_data_list = []
    is_multipage = False
    
    if output_dir is None:
        output_dir = input_path.parent / "unwrapped"
    else:
        output_dir = Path(output_dir)

    try:
        with tifffile.TiffFile(input_path) as tif:
            num_pages = len(tif.pages)
            
            is_multipage = num_pages > 1
            
            print(f"Analisi file: {input_path.name}")
            print(f" - Frames: {num_pages}")
            print(f" - Save Label: {save_label}")

            if save_label:
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f" - Salvataggio frame in: {output_dir}")

            for i, page in enumerate(tif.pages):
                
                matrix = page.asarray()
                image_data_list.append(matrix)
                
                if save_label:
                    # Costruisce un nome univoco: stackname_frame_000X.tif
                    # (Usa il nome del file originale come prefisso per evitare sovrascritture)
                    frame_name = f"{input_path.stem}_frame_{i:05d}.tif"
                    save_path = output_dir / frame_name
                    
                    tifffile.imwrite(
                        save_path, 
                        matrix, 
                        dtype=matrix.dtype, # Mantiene 16-bit se l'originale lo è
                        photometric='minisblack'
                    )
                    
                    if i % 50 == 0:
                        print(f"   ...salvato frame {i}")

    except Exception as e:
        print(f"Errore durante l'elaborazione: {e}")
        return False, []

    # 3. Output richiesti
    return is_multipage, image_data_list

# - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -#


def cut_image(image, number_of_centers, center_threshold, center_blursize, dim, off_x, off_y, img_limits, center_find_label):
    
    
    if center_find_label==True:
        center_x , center_y = center_find(image,
                                          centers = number_of_centers,          # Centro [x, y] per ritagliare l'immagine
                                          threshold = center_threshold,
                                          blursize = center_blursize)

        image = image[:, int(center_x-dim):int(center_x+dim+1),
                        int(center_y-dim):int(center_y+dim+1)]                  # Ritaglio l'immagine con i 
                                                                                # parametri appena specificati
    
    elif len(img_limits)!=0:                                                    # se specifico manualmente dim ritaglio
        image = image[:, img_limits[0]:img_limits[1], img_limits[2]:img_limits[3]]
        
    
    else:
        image = image[:, off_y:(2*dim + off_y + 1), off_x:(2*dim + off_x + 1)]

    return image

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#


def rinomina_immagini_bg(path_cartella, estensione):
       
    i = 1

    for filename in sorted(os.listdir(path_cartella)):
        if filename.endswith(estensione):
            new_filename = f"image_{i:06d}{estensione}"
            os.rename(filename, new_filename)
            i += 1
            
            
            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

def avg_image(input_path, plot_label=True, normalization=1):
    # Filtra solo file immagine (ignora .gitkeep e altri file non immagine)
    all_files = os.listdir(input_path)
    image_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif')
    images = sorted([f for f in all_files if f.lower().endswith(image_extensions)])
    
    if len(images) == 0:
        raise ValueError(f"Nessun file immagine trovato in {input_path}")
    
    N = len(images)
    x=0                                                             # inizializzo matrice normalizzata 
                                                                    # x = normalized(X)
    for i in range(N):
        ProgressBar(1 - (N - (i+1))/N)
        X = np.array(Image.open(input_path+images[i]))
        #print(X)
        if normalization == 2:
            X = (X - np.amin(X) )/(np.amax(X) - np.amin(X) )*255     # matrice i-esima normalizzata [0, 255]
        norm = np.sum(X)
        #print(norm)
        x = X/norm
        #print(np.sum(x))
        x+=x
           
    x = x / N                                                                # media
    #x = x.astype(int)                                                       # converto in int
    #img_x = Image.fromarray(x.astype("uint8"))   
    #img_x = Image.fromarray(x)      

                                                 

    if plot_label==True:                                                  
        plt.imshow(x, cmap='viridis') 
        plt.xlabel('x [pxl]')
        plt.ylabel('y [pxl]')
        plt.title("avg_image "+str(normalization))
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    _var = float(np.var(x))                                         
    _dev = float(x.std())    
    
    return x, _var, _dev            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

def avg8(input_path, plot_label=True):
    # Filtra solo file immagine (ignora .gitkeep e altri file non immagine)
    all_files = os.listdir(input_path)
    image_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif')
    images = sorted([f for f in all_files if f.lower().endswith(image_extensions)])
    
    if len(images) == 0:
        raise ValueError(f"Nessun file immagine trovato in {input_path}")
    
    N = len(images)
    som=0
    for i in range(N):
        ProgressBar(1 - (N - (i+1))/N)
        X = plt.imread(os.path.join(input_path, images[i]))
        som += X.astype(np.uint64)
           
    avg = (som / N).astype(np.uint8)              # media    

    if plot_label==True:                                                  
        plt.imshow(avg, cmap='gray') 
        plt.xlabel('x [pxl]')
        plt.ylabel('y [pxl]')
        plt.title("avg_image")
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    _var = float(np.var(avg))                                         
    _dev = float(avg.std())    
    
    return avg, _var, _dev

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


def avg16(input_path, plot_label=True):
    # Filtra solo file immagine (ignora .gitkeep e altri file non immagine)
    all_files = os.listdir(input_path)
    image_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif')
    images = sorted([f for f in all_files if f.lower().endswith(image_extensions)])
    
    if len(images) == 0:
        raise ValueError(f"Nessun file immagine trovato in {input_path}")
    
    N = len(images)
    som=0
    for i in range(N):
        # ProgressBar(1 - (N - (i+1))/N)
        X = plt.imread(os.path.join(input_path, images[i]))
        som += X.astype(np.uint64)
           
    avg = (som / N).astype(np.uint64)              # media    

    if plot_label==True:                                                  
        plt.imshow(avg, cmap='gray') 
        plt.xlabel('x [pxl]')
        plt.ylabel('y [pxl]')
        plt.title("avg_image")
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    _var = float(np.var(avg))                                         
    _dev = float(avg.std())    
    
    return avg, _var, _dev

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


def background_function(input_path, img_limits, dim, save_label=False, output_path=None, output_name="avg_image.tif", plot_label=True):
    """
    Generate average bkg image if condition is true, read average image if condition is false.
    Condition is false if: 
        1) output is already in the output directory and
        2) old_img_limits = current_img_limits
    """
    
    if output_path is None:
        output_path = input_path
        
    try:
        prev_avg_bkg_shape = plt.imread(os.path.join(output_path, output_name)).shape
        current_shape1 = (img_limits[1]-img_limits[0], img_limits[3]-img_limits[2])
        current_shape2 = (2*dim +1, 2*dim +1)
        
        cond0 = output_name in os.listdir(output_path)
        cond1 = prev_avg_bkg_shape == current_shape1
        cond2 = prev_avg_bkg_shape == current_shape2
        
        Cond =  cond0 and cond1 or cond2
    except:
        FileNotFoundError()
        Cond = False

        
    if Cond is True:
        
        img_x = plt.imread(os.path.join(output_path, output_name))   
        ProgressBar(1)
        print("Immagine media già generata.")
                  
        if plot_label==True:                                                  
            plt.imshow(img_x, cmap='viridis') 
            plt.xlabel('x [pxl]')
            plt.ylabel('y [pxl]')
            plt.title(output_name)
            plt.colorbar()
            plt.tight_layout()
            plt.show()
            plt.clf()
       
    else:
        print("Genero immagine media...")
        img_x = avg16(input_path, plot_label)
        
        plt.title(output_name)
        show_image(img_x, plot_label)                                                      
        
        
        if plot_label==True:                                                  
            plt.imshow(img_x, cmap='gray') 
            plt.xlabel('x [pxl]')
            plt.ylabel('y [pxl]')
            plt.title(output_name)
            plt.colorbar()
            plt.tight_layout()
            if save_label!=False: 
                if output_path==None:
                    raise ValueError("Specificare l'output_path quando save_label è True.")
                #plt.savefig(output_path+'avg_image.png')
                plt.imsave(os.path.join(output_path, output_name), img_x)
            plt.show()
            plt.clf()
            
    _var = float(np.var(img_x))                                         
    _dev = float(img_x.std())    
    
    
    return img_x, _var, _dev

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#


def crop_image(image, img_limits=None, dim=300, off_x=700, off_y=300):
    if img_limits!=None:                                                    # se specifico manualmente dim ritaglio
        image = image[img_limits[0]:img_limits[1], img_limits[2]:img_limits[3]]
        
    else:
        image = image[off_y:(2*dim + off_y + 1), off_x:(2*dim + off_x + 1)]

    return image
    



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

def show_image(img, save_label=False, save_path=None, output_name="show_image.png", title="show_image"):
    if save_path==None:
        save_path=""
    fig = plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.xlabel('x [pxl]')
    plt.ylabel('y [pxl]')
    plt.colorbar()
    plt.tight_layout()
    #plt.grid()
    if save_label==True: 
        #plt.imsave(os.path.join(save_path,output_name), img)
        tifffile.imwrite(os.path.join(save_path, output_name), img)
        print(f"Image by show_image() saved in {os.path.join(save_path,output_name)}")
    plt.show()
    plt.close(fig)
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

def background(bkg_path, pixel_size, medium_index, illum_wavelen, number_of_centers,
               center_threshold, center_blursize, dim, off_x, off_y, img_limits, center_find_label, 
               plot_label, save_label, save_path, save_format):
      
    
    try:
        prev_avg_bkg_shape = np.array((Image.open(bkg_path+'/avg_bg_image.tif'))).shape
        current_shape1 = (2*dim +1, 2*dim +1)
        current_shape2 = (img_limits[1]-img_limits[0], img_limits[3]-img_limits[2])
        
        cond0 = 'avg_bg_image.tif' in os.listdir(bkg_path)
        cond1 = prev_avg_bkg_shape == current_shape1
        cond2 = prev_avg_bkg_shape == current_shape2
        
        Cond =  cond0 and cond1 or cond2
    except:
        FileNotFoundError()
        Cond = False
    
    
    
    if Cond is True:
        
        # print("\n\n\nprevious avg bg shape: " + str(prev_avg_bkg_shape))
        # print("\nactual requested shape: " + str(current_shape))
                
        bkg_image = Image.open(bkg_path+'/avg_bg_image.tif')          # Open average background image
        bkg_image = np.array(bkg_image)
        var = float(np.var(bkg_image))                                          # Varianza
        dev = float(bkg_image.std())                                            # Deviazione standard

        ProgressBar(1)
          
        
    else:
        
 #       rinomina_immagini_bg(bkg_path+bkg_sub_path, ".tif")
        
        background_image_number = len(os.listdir(bkg_path))        # Getting the # of images in the directory

        bkg_image = hp.load_image(bkg_path+'/image_000001.tif',    # Carico le immagini di background
                                spacing = pixel_size,
                                medium_index = medium_index,
                                illum_wavelen = illum_wavelen)

        bkg_image = cut_image(bkg_image, number_of_centers, center_threshold,
                              center_blursize, dim, off_x, off_y, img_limits, center_find_label)
        
        #bkg_image = (bkg_image - np.amin(bkg_image) )/(np.amax(bkg_image) - np.amin(bkg_image) )*255

        norm = sum(sum(bkg_image[0, :, :]))
        
        bkg_image = bkg_image/norm
        
        
        # print("norm bkg="+str(norm))
        # print("sum bkg = "+str(sum(sum(bkg_image[0, :, :]))))


        for i in range(2, background_image_number):

            ProgressBar(1 - (background_image_number - (i+1))/background_image_number)

            if i in range(0, 10): index = '00000'+str(i)                        # Imposto il suffisso corretto per il
            elif i in range(10, 100): index = '0000'+str(i)                     # caricamento delle immagini
            elif i in range(100, 1000): index = '000'+str(i)
            elif i in range(1000, 100000): index = '00'+str(i)
            elif i in range(10000, 1000000): index = '0'+str(i)

            bkg_image_ = hp.load_image(bkg_path+'/image_'+index+'.tif',
                                    spacing = pixel_size,                       
                                    medium_index = medium_index,
                                    illum_wavelen = illum_wavelen)
            
            bkg_image_ = cut_image(bkg_image_, number_of_centers, center_threshold,
                                  center_blursize, dim, off_x, off_y, img_limits, center_find_label)
            
            #bkg_image_ = (bkg_image_ - np.amin(bkg_image_) )/(np.amax(bkg_image_) - np.amin(bkg_image_) )*255

            norm = sum(sum(bkg_image_[0, :, :]))
            
            bkg_image_ = bkg_image_/norm
            
            # print("norm bkg="+str(norm))
            # print("sum bkg = "+str(sum(sum(bkg_image[0, :, :]))))

            bkg_image += bkg_image_												

        bkg_image = bkg_image/background_image_number                           # Immagine di fondo media
       
        bkg_image = bkg_image[0, :, :]
         
        #rec_col = np.array(np.abs(_img_rec)**2/np.amax(np.abs(_img_rec)**2))
        
    
        
        var = float(np.var(bkg_image))                                          # Varianza
        dev = float(bkg_image.std())                                            # Deviazione standard

        avg = Image.fromarray(np.array(bkg_image))
        avg.save(bkg_path+'/avg_bg_image.tif')

                                                            # If True, the average background is shown
    plt.imshow(bkg_image, cmap='viridis') 
    plt.xlabel('x [pxl]')
    plt.ylabel('y [pxl]')
    plt.title('average background image')
    plt.colorbar()
    plt.tight_layout()
    if save_label==True: plt.savefig(save_path+'avg_background'+save_format)
    if plot_label==True:
        plt.show()
    plt.clf()
    
    

    return bkg_image, var, dev












