import os
import json
import base64
from datetime import datetime
import argparse

# folder and file management
from pathlib import Path
# path string repair
import ftfy

# images
from PIL import Image
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
# cleaning values of lib object for db
from decimal import Decimal

import imagehash

# video
import ffmpeg

# hash
import hashlib
#import qwen3_vl_4b

import numpy as np

# db import
import psycopg2
from psycopg2 import sql
#from dotenv import load_dotenv

from src.db_manager import DBManager

# logging
import logging
# logging config helper
from src.logging_config import LoggingConfig

# configuration loader
from src.config_load import ConfigLoader

# load configuration
config = ConfigLoader('./.env_app')

# configuration of logging
logging_config = LoggingConfig('./logs/')
logging_config.setup_logging()

#
logging.getLogger("PIL").setLevel(logging.WARNING)

import argparse

logging.info("finish init")


def file_list_from_folder_by_file_type(folder_path, valid_file_types={".jpg",}):
    # convert relativ path to absolute path
    absolute_path = os.path.abspath(folder_path)
        
    # file list
    file_list = []

    for root, dirs, files in Path(absolute_path).walk():
        for file in files:
            # Prüfe, ob die Datei eine gültige Bild-Datei ist
            if any(file.lower().endswith(ext) for ext in valid_file_types):
                # Konvertiere den Pfad in einen String
                full_path = str(root / file)
                # check encoding error
                if is_path_valid(full_path) is not True:
                    tmp_path = path_latin_to_utf8(full_path)
                    if tmp_path is None:
                        logging.error(f'error file_list_from_folder_by_file_type detect coding error by file path: {full_path}')
                        continue
                    else:
                        full_path = tmp_path
                # check exist file in index or not
                if file_exist_in_index(full_path):
                    continue
                file_list.append(full_path)
    return file_list

def calculate_image_pixel_hash(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifiziere, ob das Bild korrekt ist
            

        with Image.open(image_path) as img:
            # Konvertierung zu RGBA stellt sicher, dass Transparenz 
            # in den Hash einfließt und alle Bilder dasselbe Format haben.
            img = img.convert("RGBA")
            
            # Hash berechnen
            pixel_data = img.tobytes()
            return hashlib.sha256(pixel_data).hexdigest()
    except Exception as e:
        logging.exception(f"exception on calculate_image_pixel_hash:")
        # by over size image
        image_hash = calculate_big_image_pixel_hash(image_path)
        if image_hash:
            return image_hash
        return None

def calculate_big_image_pixel_hash(image_path):
    
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifiziere, ob das Bild korrekt ist

        with Image.open(image_path) as img:
            # Konvertiere das Bild in ein numpy-Array
            img.thumbnail((img.width // 4, img.height // 4))
            img_array = np.array(img)
            # Konvertiere zu RGBA, falls nicht bereits der Fall
            if img.mode != 'RGBA':
                img_array = np.stack([img_array] * 4, axis=-1)
            # Berechne den Hash
            return hashlib.sha256(img_array.tobytes()).hexdigest()
    except Exception as e:
        logging.exception(f"exception on calculate_big_image_pixel_hash, image path: {image_path} with message: ")
        image_hash = None
        try:
            with Image.open(image_path) as img:
                image_hash = str(imagehash.average_hash(img))
        except Exception as e:
            return image_hash
        return image_hash


'''
extract metadata from video file
'''
def get_video_metadata(video_path):
    metadata = {
        "file_path": None,
        "creation_date": None,
        'duration': None,
        'width': None,
        'height': None,
        'bit_rate': None,
        'codec_type': None,
        'codec_name': None,
        'format': None,
        'fps': None,
        'latitude': None,
        'longitude': None,
        'altitude': None
    }
    try:
        probe = ffmpeg.probe(video_path)
        video_info = probe['streams'][0]
        
        # creation time
        creation_time_str = None
        creation_time = None
        if isinstance(video_info.get('tags'), dict):
            # Suche nach 'creation_time' im Dictionary
            if 'creation_time' in video_info['tags']:
                creation_time_str = video_info['tags']['creation_time']
        elif isinstance(video_info.get('tags'), list):
            for tag in video_info['tags']:
                if isinstance(tag, dict) and 'creation_time' in tag:
                    creation_time_str = tag['creation_time']
                    break
        
        if creation_time_str:
            detected_format = detect_date_format(creation_time_str)
            if detected_format:
                try:
                    creation_time = datetime.strptime(creation_time_str, detected_format)
                except ValueError:
                    creation_time = None
        if not creation_time:
            creation_time_unix = os.path.getctime(video_path)
            creation_time = datetime.fromtimestamp(creation_time_unix)
        
        # fps and duration
        fps_str = video_info.get('r_frame_rate', '0/1')  # Default auf '0/1'
        if '/' in fps_str:
            numerator, denominator = map(int, fps_str.split('/'))
            fps = numerator / denominator
        else:
            fps = float(fps_str)
        
        #extract gps data
        latitude = None
        longitude = None
        altitude = None

        for stream in probe['streams']:
            if stream.get('tags'):
                for key, value in stream['tags'].items():
                    if 'GPS' in key:
                        # GPS-Daten konvertieren, falls notwendig
                        if isinstance(value, str) and '/' in value:
                            try:
                                numerator, denominator = map(int, value.split('/'))
                                converted_value = float(numerator / denominator)
                            except:
                                converted_value = value
                        elif isinstance(value, tuple) and len(value) == 2:
                            converted_value = float(value[0] / value[1])
                        else:
                            converted_value = value

                        # Speichere nur die relevanten GPS-Daten
                        if key == 'GPSLatitude':
                            latitude = converted_value
                        elif key == 'GPSLongitude':
                            longitude = converted_value
                        elif key == 'GPSAltitude':
                            altitude = converted_value

        
        #'filename': os.path.basename(video_path),
        metadata['file_path'] = video_path
        metadata['creation_date'] = creation_time
        metadata['duration'] = float(video_info.get('duration', 'N/A'))
        metadata['width'] = video_info.get('width', 'N/A')
        metadata['height'] = video_info.get('height', 'N/A')
        metadata['bit_rate'] = video_info.get('bit_rate', 'N/A')
        metadata['codec_type'] = video_info.get('codec_type', 'N/A')
        metadata['codec_name'] = video_info.get('codec_name', 'N/A')
        metadata['format'] = os.path.splitext(video_path)[1]
        metadata['fps'] = fps
        metadata['latitude'] = latitude
        metadata['longitude'] = longitude
        metadata['altitude'] = altitude
        
        return metadata 
    except Exception as e:
        logging.exception(f"exception on get_video_metadata, video path: {video_path} with message: ")
        return None

def detect_date_format(date_str):
    # Liste der gängigen Datumformate, die getestet werden sollen
    formats = [
        "%Y:%m:%d %H:%M:%S",  # z. B. "2018:08:26 17:08:32"
        "%Y-%m-%d %H:%M:%S",  # z. B. "2018-08-26 17:08:32"
        "%d.%m.%Y %H:%M:%S",  # z. B. "26.08.2018 17:08:32"
        "%Y:%m:%d %H:%M:%S",  # z. B. "2018:08:26 17:08:32"
        "%Y:%m:%d %H:%M:%S.",  # z. B. "2018:08:26 17:08:32."
        "%Y:%m:%d %H:%M:%S,%f",  # z. B. "2018:08:26 17:08:32,000"
        "%d/%m/%Y %H:%M:%S",  # z. B. "26/08/2018 17:08:32"
        "%Y/%m/%d %H:%M:%S",  # z. B. "2018/08/26 17:08:32"
        "%Y-%m-%d",            # z. B. "2018-08-26"
        "%d.%m.%Y",            # z. B. "26.08.2018"
        "%d/%m/%Y",            # z. B. "26/08/2018"
        "%Y/%m/%d",            # z. B. "2018/08/26"
        "%Y-%m-%dT%H:%M:%S.%fZ",  # z. B. "2018-08-03T15:37:58.000000Z"
    ]

    for fmt in formats:
        try:
            datetime.strptime(date_str, fmt)
            return fmt
        except ValueError:
            continue

    return None  # Kein passendes Format gefunden

def get_clean_gps_coords(gps_info):
    """
    Eingabe: Der gps_info-Block aus Pillow EXIF.
    Ausgabe: (lat, lon, alt) als Dezimal-Floats oder None.
    """
    if not gps_info:
        return None, None, None

    def to_decimal(values, ref):
        if not values or len(values) < 3:
            return None
        # Pillow IFDRational in Float umwandeln und DMS zu Dezimalgrad berechnen
        degrees = float(values[0])
        minutes = float(values[1])
        seconds = float(values[2])
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal

    # IDs für GPS-Tags innerhalb des GPS-Blocks
    # 1: Ref, 2: Lat | 3: Ref, 4: Lon | 5: AltRef, 6: Alt
    lat = to_decimal(gps_info.get(2), gps_info.get(1))
    lon = to_decimal(gps_info.get(4), gps_info.get(3))
    
    # Altitude Logik
    alt = gps_info.get(6)
    if alt is not None:
        alt = float(alt)
        alt_ref = gps_info.get(5)
        # In EXIF bedeutet AltRef 1 'unter dem Meeresspiegel'
        if alt_ref == 1 or (isinstance(alt_ref, bytes) and int.from_bytes(alt_ref, 'big') == 1):
            alt = -alt

    return lat, lon, alt

def get_image_metadata(image_path):
    metadata = {
        "file_path": image_path,
        "creation_date": None,
        "latitude": None,
        "longitude": None,
        "altitude": None,
        "image_width": None,
        "image_height": None,
        "image_format": None,
        "image_mode": None        
    }
    TAG_NAME_TO_ID = {v: k for k, v in TAGS.items()}
    
    try:
        with Image.open(image_path) as img:
            
            # EXIF-Daten extrahieren
            exif_data = img._getexif()
            if exif_data is None:
                return metadata
            
            # image property
            metadata['image_width'], metadata['image_height'] = img.size
            metadata['image_format'] = img.format
            metadata['image_mode'] = img.mode

            # extrac createtime
            dt_tag_id = TAG_NAME_TO_ID.get('DateTimeOriginal')
            creation_date_str = exif_data.get(dt_tag_id)
            
            if creation_date_str:
                detected_format = detect_date_format(creation_date_str)
                if detected_format:
                    try:
                        metadata['creation_date'] = datetime.strptime(creation_date_str, detected_format)
                    except ValueError:
                        metadata['creation_date'] = None

            # GPS-Data extract
            gps_tag_id = TAG_NAME_TO_ID.get('GPSInfo')
            gps_info_raw = exif_data.get(gps_tag_id)
            
            if gps_info_raw:
                gps_clean_data = get_clean_gps_coords(gps_info_raw)
                metadata['latitude'] = gps_clean_data[0]
                metadata['longitude'] = gps_clean_data[1]
                metadata['altitude'] = gps_clean_data[2]
            return metadata

    except Exception as e:
        logging.exception(f"error on get_image_metadata with message: {e}")
        return metadata

'''
extract frame as picture and generate hash
'''
def extract_keyframes(video_path, output_dir , output_pattern='image_%02d.png' , num_images=3, metadata=None):
    import math
    # create temp folder
    os.makedirs(output_dir, exist_ok=True)

    # calulate frames by fps and duration   
    total_frames = int(metadata['fps'] * (round(metadata['duration'],0)))
    frame_positions = [int(total_frames / num_images * i) for i in range(1, num_images + 1)]

    # Extrahiereextract thumbnails
    thumbnails = {}
    for i, frame_number in enumerate(frame_positions):
        output_path = output_pattern % (i + 1)
        # create thumbnail from video
        (
            ffmpeg
            .input(video_path)
            .filter('select', f'eq(n,{frame_number})')
            .output(output_dir+output_path, vframes=1, format='image2', pix_fmt='rgb24')
            .run(overwrite_output=True)
        )
        if os.path.exists(output_dir+output_path):
            # generate hash from thumbnails
            image_hash = calculate_image_pixel_hash(output_dir+output_path)
        else:
            continue
        thumbnails[frame_number]=image_hash
        os.remove(output_dir+output_path)
    os.removedirs(output_dir)
    return thumbnails


# ToDo problem mit pfaden lösen 
def is_path_valid(path):
    if ftfy.badness.is_bad(path):
        return True
    return False

def path_latin_to_utf8(path):
    
    if isinstance(path, bytes):
        # Wenn der Pfad als Bytes vorliegt, versuche, ihn mit 'utf-8' zu decodieren
        # Falls das nicht geht, versuche mit 'latin-1'
        try:
            tmp_path = path.decode('utf-8')
        except UnicodeDecodeError:
            tmp_path = path.decode('latin-1')
    elif isinstance(path, str):
        tmp_path = path
    else:
        # Wenn der Pfad weder Bytes noch String ist, werfe einen Fehler
        raise ValueError("Ungültiger Pfad-Typ. Erwartet wurden 'str' oder 'bytes'.")
    
    if not ftfy.badness.is_bad(tmp_path):
        return tmp_path        
    return None

def create_image_index_from_folder(folder_path):
    
    valid_file_types = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    image_files = file_list_from_folder_by_file_type(folder_path, valid_file_types)

    batch_size = 100
    '''
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        save_index(batch)
    '''
    
    image_list = []
    for image_file in image_files:
        hash_value = calculate_image_pixel_hash(image_file)
        metadata = get_image_metadata(image_file)
        image_list.append((image_file, {'0':hash_value}, metadata))  # Tupel statt Dictionary
        if len(image_list)<batch_size:
            continue
        save_index(image_list)
        for i in range(0, len(image_list),1):
            save_image_metadata(image_list[i][2])

        image_list = None
        image_list = []

    

def create_video_index_from_folder(folder_path):
    
    valid_file_types = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv"}
    video_files = file_list_from_folder_by_file_type(folder_path, valid_file_types)

    video_list = []
    for video_file in video_files:
        # extract metadata of video file
        metadata = get_video_metadata(video_file)
        if metadata is None:
            continue

        # extract frames from video and generate hashes
        thumbnail_hash = extract_keyframes( video_file,
                                        folder_path+'/tmp/',
                                        'image_%02d.png',
                                        4,
                                        metadata
                                    )
        # add path, thumbnail_hash, metadata 
        video_list.append((video_file, thumbnail_hash, metadata))
    return video_list


######## system helper
def save_index(batch):

    try:
        db_conf = config.get_db_config()
        db_manager = DBManager(db_conf)

        # Batch-Insert
        insert_query = sql.SQL("INSERT INTO tblFileIndex (file_path, pixel_hash) VALUES (%s, %s) ON CONFLICT (file_path) DO NOTHING;")
        data_to_insert = [(entry[0], json.dumps(entry[1])) for entry in batch]
        db_manager.executemany_sql(insert_query, data_to_insert)

        logging.info(f'insert {len(batch)} entrys into tblFileIndex')
    except Exception as e:
        logging.exception(f"exception on save_image_index, with message: ")

def save_video_metadata(metadata):
    conn = None
    try:
        db_conf = config.get_db_config()
        db_manager = DBManager(db_conf)

        # Batch-Insert
        insert_query = sql.SQL("""INSERT INTO tblvideometadata (
            file_path, duration, width, height, bit_rate, 
            codec_type, codec_name, format, fps, creation_date,
            latitude, longitude, altitude                   
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
        ON CONFLICT (file_path) DO NOTHING;""")

        batch = (
            metadata.get('file_path'),
            metadata.get('duration'),
            metadata.get('width'),
            metadata.get('height'),
            metadata.get('bit_rate'),

            metadata.get('codec_type'),
            metadata.get('codec_name'),
            metadata.get('format'),
            metadata.get('fps'),
            metadata.get('creation_date'),
            
            metadata.get('latitude'),
            metadata.get('longitude'),
            metadata.get('altitude'),

        )

        db_manager.execute_sql(insert_query, batch)

        print(f"{len(batch)} Video-Metadaten erfolgreich gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der Video-Metadaten: {e}")

def file_exist_in_index(file_path):
    try:
        db_conf = config.get_db_config()
        db_manager = DBManager(db_conf)
        
        query = """
            SELECT EXISTS (
                SELECT 1
                FROM tblFileIndex
                WHERE file_path = %s
            );
        """
        # Prüfe, ob der Dateipfad bereits existiert
        rep_data = db_manager.execute_sql(query, file_path, 1) # fetchone
        if not rep_data:
            raise ValueError(f"{__name__}:empty result not allowed")
        return rep_data[0]
    except Exception as e:
            logging.exception(f"exception on file_exist_in_index, with message: ")
    
def save_image_metadata(metadata):
    try:
        db_conf = config.get_db_config()
        db_manager = DBManager(db_conf)
        
        # SQL-Abfrage zum Einfügen der Metadaten
        # Hinweis:  latitude, longitude, altitude referenzen entfernt aktuell nicht wichtig, spartial kann später brechnet werden für umkreis berechnung
        insert_query = """
        INSERT INTO tblimagemetadata (
            file_path, creation_date, latitude, longitude, altitude,
            image_width, image_height, image_format, image_mode
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
        """
        
        # Werte vorbereiten und Typen sicherstellen
        values = (
            metadata.get('file_path'),
            metadata.get('creation_date'),
            metadata.get('latitude'),
            metadata.get('longitude'),
            metadata.get('altitude'),
            metadata.get('image_width'),
            metadata.get('image_height'),
            metadata.get('image_format'),
            metadata.get('image_mode'),
        )

        db_manager.execute_sql(insert_query, values)
    except Exception as e:

        print(f"error save_image_metadata with message: {e}")

########

def get_parameter():
    parser = argparse.ArgumentParser(description="image scanner: scan all media files into the path")

    
    parser.add_argument("--scan-path", required=True, help="path to scan")
    parser.add_argument("--output", required=False, help="path for selected files")
    #parser.add_argument("--file-typ", action="append", help="find specificate file type")
    parser.add_argument("--log-level", required=False, help="set logging level, debug")
    
    parser.add_argument("--with-db", action="store_true", help="for using of postgres db")

    # Parse die Argumente
    args = parser.parse_args()
    
    # Verwende die Parameter
    #print(f"Input: {args.input}")
    #print(f"Output: {args.output}")
    return args

####### dev helper
def is_valid_project_table_name(table_name):
    # Einfache Prüfung: nur Buchstaben, Zahlen und Unterstriche erlaubt
    if not table_name.isalnum() or not table_name.islower() or "_" not in table_name:
        return False
    return True

def delete_all_entries_from_table(table_name):
    conn = None
    try:
        # whitelist for project tables
        #if not is_valid_project_table_name(table_name):
        #    logging.warning(f"Ungültiger Tabellenname: {table_name}")
        #    return
        
        # load db config and prepare query
        db_conf = config.get_db_config()
        db_manager = DBManager(db_conf)
        query = sql.SQL("DELETE FROM {table}").format(table=sql.Identifier(table_name))
        # execute query
        db_manager.execute_sql(query)
        logging.debug(f'delete all entrys from {table_name}')

    except Exception as e:
        logging.exception(f"exception on delete_all_entries_from_table, with message: ")


###### face rec
import os
import cv2
import numpy as np
from deepface import DeepFace

# Konfiguriere DeepFace für CPU-Betrieb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def extrahiere_gesichter_und_features(bild_pfad):
        """
        Führt Gesichtsanalyse, Ausrichtung, Extraktion und Feature-Extraktion durch.

        Args:
            bild_pfad (str): Der Pfad zum Eingabebild.

        Returns:
            list: Eine Liste von Dictionaries, die pro Gesicht die 
                Feature-Vektoren (Embedding), das zugeschnittene Gesicht (als numpy array) 
                und die Koordinaten (Box) enthalten.
        """
        ergebnisse = []

        try:
            # Schritt 1: Gesichtserkennung, Lokalisierung und Ausrichtung
            # Nutzt 'retinaface' als robusten Detektor für gute Ergebnisse
            detektionen = DeepFace.extract_faces(
                img_path=bild_pfad, 
                detector_backend='retinaface', 
                enforce_detection=False, # Analysiert auch Bilder ohne 100% sicheres Gesicht
                align=True # Führt die Ausrichtung automatisch durch
            )
            
            # Schritt 2 & 3: Feature-Extraktion
            for gesicht_objekt in detektionen:
                # Das zugeschnittene, ausgerichtete Gesicht liegt als NumPy-Array vor
                zugeschnittenes_gesicht_rgb = gesicht_objekt['face']
                
                # DeepFace.represent generiert die 512-dimensionalen Vektoren (Features)
                # Wir nutzen Facenet512 für hohe Genauigkeit
                features = DeepFace.represent(
                    img_path=zugeschnittenes_gesicht_rgb,
                    model_name="Facenet512",
                    enforce_detection=False # Da Gesichter schon extrahiert wurden
                )

                # Sammle alle relevanten Daten
                ergebnisse.append({
                    'embedding': features[0]['embedding'], # Der Feature-Vektor
                    'gesicht_bild_daten': zugeschnittenes_gesicht_rgb, # Das zugeschnittene Gesicht (RGB)
                    'box_koordinaten': gesicht_objekt['box'], # Die Original-Koordinaten im Bild
                    'bild_pfad': bild_pfad
                })
                
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {bild_pfad}: {e}")

        return ergebnisse




def findDuplicateImages(folder_path):
    valid_file_types = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    image_files = file_list_from_folder_by_file_type(folder_path, valid_file_types)

    image_dup_list = set()
    image_list = set()
    for image_file in image_files:
        hash_value = calculate_image_pixel_hash(image_file)        
        if hash_value in image_list:
            image_dup_list.append(image_file)
        else:
            image_list.append(hash_value)

def findDuplicateVideos(folder_path):
    valid_file_types = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv"}
    video_files = file_list_from_folder_by_file_type(folder_path, valid_file_types)
    video_list = set()
    video_dub_list = set()

    for video_file in video_files:
        # extract metadata of video file
        metadata = get_video_metadata(video_file)
        if metadata is None:
            continue

        # extract frames from video and generate hashes
        thumbnail_hash = extract_keyframes( video_file,
                                        folder_path+'/tmp/',
                                        'image_%02d.png',
                                        4,
                                        metadata
                                    )
        # add path, thumbnail_hash, metadata 
        if thumbnail_hash in video_list:
            video_dub_list.append(video_file)
        else:
            video_list.append(thumbnail_hash)

def main():
    
    args = get_parameter()
    
    if args.with_db:
       # load config
        logging.info("server mode: start init")
        # db connection check
        db_conf = config.get_db_config()
        db_manager = DBManager(db_conf)
        logging.info("server mode: finish init")

    # ToDo:
    # - speichern der video hashes , erledigt
    # - abgleich der doppelten dateien (view existieren)
    # - merge function der dateien
    # - sortieren in ordner
    # weitere Funktionen
    # - auslesen von gps daten (BIlder erledigt, Videos muss noch getestet werden), lokalisierung der Bilder nach Motiven
    # - kategorisierung der Aufnahmen nach Personen
    # - beschreibung der fotos
    
    folder_path = args.scan_path
    
    
    #folder_path = './img_example/'
    #folder_path = './img_short_example/'
    #folder_path = './example/db_fam/'
    #folder_path = './gps_example/'
    #folder_path = './video_example'
    #folder_path = '/run/media/pkoehler/a605c951-e6e6-4aaf-967c-e5589cb19fb81/data/merge_files'
    #folder_path = r'file:///run/media/pkoehler/a605c951-e6e6-4aaf-967c-e5589cb19fb81/data/Handb%FCcher%20Quelltexte%20Erl%E4uterungen'
    #folder_path = '/run/media/pkoehler/a605c951-e6e6-4aaf-967c-e5589cb19fb81/data/bilder unsortiert'
    #folder_path = '/run/media/pkoehler/a605c951-e6e6-4aaf-967c-e5589cb19fb81/data/bilder_auto_sort/'
    #folder_path = '/run/media/pkoehler/a605c951-e6e6-4aaf-967c-e5589cb19fb81/data'
    
    # check encoding error
    if is_path_valid(folder_path) is not True:
        tmp_path = path_latin_to_utf8(folder_path)
        if tmp_path is None:
            logging.error(f'error main detect coding error by file path: {folder_path}')
            #continue
        else:
            folder_path = tmp_path
    if os.path.exists(folder_path) == False:
        print('Path not exist')
        return 1



    ######## DEBUG ########
    # reset db
    #delete_all_entries_from_table("tblfileindex")
    #delete_all_entries_from_table("tblimagemetadata")
    #delete_all_entries_from_table("tblvideometadata")

    ######## PROCESS ########
    # search img files and create hash
    image_list = create_image_index_from_folder(folder_path)
    
    # save saerch results from image list
    
    print(f"finish save metadata from files ")
    
    # saerch video files and create hashes
    video_list = create_video_index_from_folder(folder_path)   
    # save saerch results from image list
    video_batch_size = 100
    for i in range(0, len(video_list), video_batch_size):
        video_batch = video_list[i:i + video_batch_size]
        save_index(video_batch)
    
    for i in range(0, len(video_list),1):
        save_video_metadata(video_list[i][2])

    #for i in range(0, len(video_list), video_batch_size):
    #    video_batch = video_list[i:i + video_batch_size]
    #    save_video_metadata(video_batch)
    '''
    from deepface import DeepFace
    from sklearn.cluster import DBSCAN
    import numpy as np

    # 1. Alle Gesichter in Vektoren umwandeln
    embeddings = []
    for image in image_list:
        face_obj = DeepFace.represent(img_path=image[0], model_name="Facenet512")[0]
        embeddings.append(face_obj["embedding"])

    # 2. Gruppieren (DBSCAN erkennt die Anzahl der Personen selbst)
    # 'eps' steuert die Empfindlichkeit (kleiner = strenger)
    clustering_model = DBSCAN(eps=0.5, metric="cosine", min_samples=1)
    labels = clustering_model.fit_predict(embeddings)
    

    '''

    '''
    
    # Beispielaufruf und Anzeige der Ergebnisse
    bild = "./urlaub_2026.jpg" # Pfad zu Ihrem Bild
    analysedaten = extrahiere_gesichter_und_features(bild)

    print(f"\nIm Bild {bild} wurden {len(analysedaten)} Gesichter gefunden.")

    for i, daten in enumerate(analysedaten):
        print(f"--- Gesicht {i+1} ---")
        print(f"Feature-Vektor-Länge: {len(daten['embedding'])}")
        print(f"Koordinaten (Box): {daten['box_koordinaten']}")
        # print(f"Gesicht als Numpy-Array: {daten['gesicht_bild_daten'].shape}")
        
        # Optional: Das extrahierte Gesicht anzeigen oder speichern
        # cv2.imshow(f"Gesicht {i+1}", cv2.cvtColor(daten['gesicht_bild_daten'], cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        return True
    '''
if __name__ == "__main__":
    main()