-- Azure Pricing Tables
-- Creates tables to store Azure SKU pricing information

-- VM Pricing Table
CREATE TABLE IF NOT EXISTS __schema__.azure_pricing_vm (
    id SERIAL PRIMARY KEY,
    sku_name VARCHAR(255),
    product_name VARCHAR(255),
    arm_sku_name VARCHAR(255),
    arm_region_name VARCHAR(100),
    retail_price DECIMAL(18, 6),
    unit_price DECIMAL(18, 6),
    currency_code VARCHAR(10),
    unit_of_measure VARCHAR(50),
    meter_name VARCHAR(255),
    type VARCHAR(100),
    is_primary_meter_region BOOLEAN,
    effective_start_date TIMESTAMP,
    last_updated TIMESTAMP,
    pricing_tier VARCHAR(50)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_azure_pricing_vm_sku
    ON __schema__.azure_pricing_vm(sku_name, arm_region_name);

CREATE INDEX IF NOT EXISTS idx_azure_pricing_vm_region
    ON __schema__.azure_pricing_vm(arm_region_name);


-- Storage Pricing Table
CREATE TABLE IF NOT EXISTS __schema__.azure_pricing_storage (
    id SERIAL PRIMARY KEY,
    sku_name VARCHAR(255),
    product_name VARCHAR(255),
    arm_region_name VARCHAR(100),
    retail_price DECIMAL(18, 6),
    unit_price DECIMAL(18, 6),
    currency_code VARCHAR(10),
    unit_of_measure VARCHAR(50),
    meter_name VARCHAR(255),
    type VARCHAR(100),
    effective_start_date TIMESTAMP,
    last_updated TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_azure_pricing_storage_sku
    ON __schema__.azure_pricing_storage(sku_name, arm_region_name);


-- Managed Disk Pricing Table
CREATE TABLE IF NOT EXISTS __schema__.azure_pricing_disk (
    id SERIAL PRIMARY KEY,
    sku_name VARCHAR(255),
    product_name VARCHAR(255),
    arm_region_name VARCHAR(100),
    retail_price DECIMAL(18, 6),
    unit_price DECIMAL(18, 6),
    currency_code VARCHAR(10),
    unit_of_measure VARCHAR(50),
    meter_name VARCHAR(255),
    effective_start_date TIMESTAMP,
    last_updated TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_azure_pricing_disk_sku
    ON __schema__.azure_pricing_disk(sku_name, arm_region_name);


-- Public IP Pricing Table
CREATE TABLE IF NOT EXISTS __schema__.azure_pricing_ip (
    id SERIAL PRIMARY KEY,
    sku_name VARCHAR(255),
    product_name VARCHAR(255),
    arm_region_name VARCHAR(100),
    retail_price DECIMAL(18, 6),
    unit_price DECIMAL(18, 6),
    currency_code VARCHAR(10),
    unit_of_measure VARCHAR(50),
    meter_name VARCHAR(255),
    effective_start_date TIMESTAMP,
    last_updated TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_azure_pricing_ip_sku
    ON __schema__.azure_pricing_ip(sku_name, arm_region_name);


-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA __schema__ TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA __schema__ TO PUBLIC;
