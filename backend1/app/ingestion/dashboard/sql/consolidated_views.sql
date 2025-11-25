CREATE OR REPLACE VIEW __dashboardname__.view_fact_billing AS
SELECT 
    f.hash_key,                           -- PK
    f.cloud_source,
    f.billedcost,
    f.effectivecost,
    f.contractedcost,
    f.listcost,
    f.consumedquantity,
    f.billingperiodstart,                 -- FK
    f.billingcurrency,                    
    f.chargedescription,                 
    f.chargeclass,                       
    f.commitmentdiscountcategory,         
    f.commitmentdiscountid,              
    f.commitmentdiscountname,            
    f.subaccountid,                      
    f.regionid,                           -- FK
    f.providername,                       -- FK
    f.resourceid,                         -- FK
    f.servicename,                        -- FK
    f.pricingcategory,                    -- FK
    __budget__ :: integer AS monthly_budget
FROM 
    __dashboardname__.target_table f;

-- Create View for Dim_Time
CREATE OR REPLACE VIEW __dashboardname__.view_dim_time AS
SELECT DISTINCT 
    billingperiodstart,
    billingperiodend,
    chargeperiodstart,
    chargeperiodend
FROM 
    __dashboardname__.target_table;

-- Create View for Dim_Region
CREATE OR REPLACE VIEW __dashboardname__.view_dim_region AS
SELECT DISTINCT 
    regionid,
    regionname
FROM 
    __dashboardname__.target_table;

-- Create View for Dim_Provider
CREATE OR REPLACE VIEW __dashboardname__.view_dim_provider AS
SELECT DISTINCT 
    providername,
    publishername
FROM 
    __dashboardname__.target_table;

-- Create View for Dim_Resource
CREATE OR REPLACE VIEW __dashboardname__.view_dim_resource AS
SELECT DISTINCT 
    resourceid,
    resourcename,
    resourcetype
FROM 
    __dashboardname__.target_table;

-- Create View for Dim_Service
CREATE OR REPLACE VIEW __dashboardname__.view_dim_service AS
SELECT DISTINCT 
    servicename,
    servicecategory,
    skuid,
    skupriceid
FROM 
    __dashboardname__.target_table;

-- Create View for Dim_Pricing
CREATE OR REPLACE VIEW __dashboardname__.view_dim_pricing AS
SELECT DISTINCT 
    pricingcategory,
    pricingunit,
    contractedunitprice,
    listunitprice,
    pricingquantity
FROM 
    __dashboardname__.target_table;