CREATE TABLE IF NOT EXISTS tblFileIndex (
    file_path TEXT NOT NULL PRIMARY KEY,
    pixel_hash JSONB NOT NULL,
    scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    --- create_time
);


'''
view for all files with hash values
'''
--DROP VIEW vw_files_with_pixel_hashes;
CREATE OR REPLACE VIEW vw_files_with_pixel_hashes AS
SELECT 
    file_path,
    jsonb_object_keys(pixel_hash) AS frame_id,
    jsonb_extract_path_text(pixel_hash, jsonb_object_keys(pixel_hash)) AS hash_value
FROM 
    tblFileIndex
WHERE 
    pixel_hash IS NOT NULL
    AND pixel_hash <> '{}';
--SELECT * FROM vw_files_with_pixel_hashes;


--DROP VIEW vw_files_dupicate_like_pixel_hashes;
CREATE OR REPLACE VIEW vw_files_dupicate_like_pixel_hashes AS
SELECT 
    o_data.file_path as o_file_path,
	c_data.file_path as c_file_path,
    o_data.frame_id as o_frame_id,
    c_data.frame_id as c_frame_id,
    o_data.hash_value as o_hash_value,
    c_data.hash_value as c_hash_value
FROM 
    vw_files_with_pixel_hashes AS o_data,
    vw_files_with_pixel_hashes AS c_data
WHERE 
    o_data.hash_value = c_data.hash_value
	AND o_data.file_path != c_data.file_path
	AND o_data.file_path < c_data.file_path;	
--SELECT * FROM vw_files_dupicate_like_pixel_hashes;


CREATE TABLE tblimagemetadata (
    id SERIAL PRIMARY KEY,
    
    -- Datei- und Systeminfos
    file_path TEXT NOT NULL,
    creation_date TIMESTAMP WITHOUT TIME ZONE,
    
    -- Bildeigenschaften
    image_width INTEGER,
    image_height INTEGER,
    image_format VARCHAR(10),
    image_mode VARCHAR(10),
    
    -- GPS-Koordinaten (Dezimalgrad)
    -- DECIMAL(10, 7) ermöglicht eine Präzision von ca. 1 cm
    latitude DECIMAL(10, 7),
    longitude DECIMAL(11, 7),
    latitude_ref CHAR(1),  -- 'N' oder 'S'
    longitude_ref CHAR(1), -- 'E' oder 'W'
    
    -- Höhe (in Metern)
    altitude DOUBLE PRECISION,
    altitude_ref SMALLINT, -- 0 = über Meeresspiegel, 1 = unter Meeresspiegel
    
    -- Flexibler Speicher für alle restlichen EXIF-Tags
    -- JSONB ist schneller bei Abfragen und unterstützt Indizes
    exif_json JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indizes für schnelle Suche
CREATE INDEX idx_image_creation_date ON tblimagemetadata(creation_date);
CREATE INDEX idx_image_location ON tblimagemetadata(latitude, longitude);






CREATE TABLE IF NOT EXISTS tblvideometadata (
    file_path TEXT NOT NULL PRIMARY KEY,
    duration FLOAT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    bit_rate TEXT NOT NULL,
    codec_type TEXT NOT NULL,
    codec_name TEXT NOT NULL,
    format TEXT NOT NULL,
    fps FLOAT NOT NULL,
    creation_date TIMESTAMP WITHOUT TIME ZONE,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    latitude DECIMAL(10, 7),
    longitude DECIMAL(11, 7),
    latitude_ref CHAR(1),  -- 'N' oder 'S'
    longitude_ref CHAR(1), -- 'E' oder 'W'
    
    -- Höhe (in Metern)
    altitude DOUBLE PRECISION,
    altitude_ref SMALLINT -- 0 = über Meeresspiegel, 1 = unter Meeresspiegel
);
CREATE INDEX idx_image_creation_date ON tblvideometadata(creation_date);
CREATE INDEX idx_image_location ON tblvideometadata(latitude, longitude);