-- drop schema IF EXISTS __schema__ cascade;

create schema if not exists __schema__;

-- Ensure the role exists and grant the required privileges directly
DO
$$
BEGIN
    -- Create role if it does not exist
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cloudmeterpsqladmin') THEN
        EXECUTE 'CREATE ROLE cloudmeterpsqladmin WITH LOGIN PASSWORD ''__password__''';
    END IF;

    -- Grant privileges to the role
    IF EXISTS (SELECT 1 FROM pg_database WHERE datname = '__databasename__') THEN
        EXECUTE 'GRANT ALL PRIVILEGES ON DATABASE "__databasename__" TO cloudmeterpsqladmin';
    END IF;

    -- Grant privileges to the role on schema and existing objects
    EXECUTE 'GRANT USAGE, CREATE ON SCHEMA public TO cloudmeterpsqladmin';
    EXECUTE 'GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cloudmeterpsqladmin';
    EXECUTE 'GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cloudmeterpsqladmin';
    EXECUTE 'GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO cloudmeterpsqladmin';
END
$$;
