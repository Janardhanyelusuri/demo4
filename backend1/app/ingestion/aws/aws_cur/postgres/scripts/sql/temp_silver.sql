ALTER TABLE silver.aws_cur_standard
ADD COLUMN tags JSONB;

-- Create a temporary table to hold JSONB values
CREATE TEMP TABLE temp_tags (tags JSONB);

-- Insert unique JSONB values into the temporary table
INSERT INTO temp_tags (tags) VALUES
('{"team": "devops", "product": "cm"}'::jsonb),
('{"team": "marketing", "product": "crm"}'::jsonb),
('{"team": "finance", "product": "erp"}'::jsonb),
('{"team": "hr", "product": "workday"}'::jsonb),
('{"team": "engineering", "product": "jira"}'::jsonb),
('{"team": "support", "product": "zendesk"}'::jsonb),
('{"team": "design", "product": "figma"}'::jsonb),
('{"team": "legal", "product": "clio"}'::jsonb),
('{"team": "it", "product": "service_now"}'::jsonb),
('{"team": "security", "product": "splunk"}'::jsonb),
('{"team": "product", "product": "productboard"}'::jsonb),
('{"team": "research", "product": "qualtrics"}'::jsonb),
('{"team": "growth", "product": "hubspot"}'::jsonb),
('{"team": "media", "product": "sprinklr"}'::jsonb),
('{"team": "data", "product": "snowflake"}'::jsonb);

WITH row_numbers AS (
    SELECT ctid, ROW_NUMBER() OVER () AS rn
    FROM silver.aws_cur_standard
),
tag_rows AS (
    SELECT tags, ROW_NUMBER() OVER () AS tag_rn
    FROM temp_tags
),
all_tags AS (
    SELECT ctid, rn, tags
    FROM row_numbers
    JOIN tag_rows
    ON (rn - 1) % (SELECT COUNT(*) FROM temp_tags) + 1 = tag_rn
)
UPDATE silver.aws_cur_standard
SET tags = all_tags.tags
FROM all_tags
WHERE silver.aws_cur_standard.ctid = all_tags.ctid;
