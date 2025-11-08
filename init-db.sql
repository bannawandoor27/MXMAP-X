-- Create database if it doesn't exist
SELECT 'CREATE DATABASE mxmap_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mxmap_db')\gexec
