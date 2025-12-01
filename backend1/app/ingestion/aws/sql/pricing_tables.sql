-- AWS Pricing Tables
-- Creates tables to store AWS service pricing information

-- EC2 Instance Pricing Table
CREATE TABLE IF NOT EXISTS __schema__.aws_pricing_ec2 (
    id SERIAL PRIMARY KEY,
    instance_type VARCHAR(100),
    vcpu VARCHAR(50),
    memory VARCHAR(50),
    storage VARCHAR(100),
    network_performance VARCHAR(100),
    instance_family VARCHAR(100),
    physical_processor VARCHAR(255),
    clock_speed VARCHAR(50),
    price_per_hour DECIMAL(18, 6),
    currency VARCHAR(10),
    region VARCHAR(50),
    region_name VARCHAR(100),
    operating_system VARCHAR(50),
    tenancy VARCHAR(50),
    unit VARCHAR(50),
    last_updated TIMESTAMP
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_aws_pricing_ec2_instance_type
    ON __schema__.aws_pricing_ec2(instance_type, region);

CREATE INDEX IF NOT EXISTS idx_aws_pricing_ec2_region
    ON __schema__.aws_pricing_ec2(region);

CREATE INDEX IF NOT EXISTS idx_aws_pricing_ec2_family
    ON __schema__.aws_pricing_ec2(instance_family);


-- S3 Storage Pricing Table
CREATE TABLE IF NOT EXISTS __schema__.aws_pricing_s3 (
    id SERIAL PRIMARY KEY,
    storage_class VARCHAR(100),
    volume_type VARCHAR(100),
    usage_type VARCHAR(100),
    price_per_unit DECIMAL(18, 6),
    currency VARCHAR(10),
    region VARCHAR(50),
    region_name VARCHAR(100),
    unit VARCHAR(50),
    description TEXT,
    last_updated TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_aws_pricing_s3_storage_class
    ON __schema__.aws_pricing_s3(storage_class, region);

CREATE INDEX IF NOT EXISTS idx_aws_pricing_s3_region
    ON __schema__.aws_pricing_s3(region);


-- EBS Volume Pricing Table
CREATE TABLE IF NOT EXISTS __schema__.aws_pricing_ebs (
    id SERIAL PRIMARY KEY,
    volume_type VARCHAR(100),
    storage_media VARCHAR(50),
    max_iops_volume VARCHAR(50),
    max_throughput_volume VARCHAR(50),
    price_per_gb_month DECIMAL(18, 6),
    currency VARCHAR(10),
    region VARCHAR(50),
    region_name VARCHAR(100),
    unit VARCHAR(50),
    last_updated TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_aws_pricing_ebs_volume_type
    ON __schema__.aws_pricing_ebs(volume_type, region);

CREATE INDEX IF NOT EXISTS idx_aws_pricing_ebs_region
    ON __schema__.aws_pricing_ebs(region);


-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA __schema__ TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA __schema__ TO PUBLIC;
