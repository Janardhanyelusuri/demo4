-- Function to extract column from product
CREATE OR REPLACE FUNCTION public.extract_col_from_product(
    product text,
    col text)
    RETURNS text
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
DECLARE
    n integer := number_of_chars(product, ')');
    i integer := 1;
    e text := '';
BEGIN
    LOOP  
        EXIT WHEN i - 1 = n;
        IF split_part(product, ')', i) LIKE concat('%', col, '%') THEN
           e := remove_chars(split_part(product, ')', i), array['"', '\', ',', '(', ')', '{', col]);
        END IF;
        i := i + 1;
    END LOOP; 
    RETURN e;
END;
$BODY$;

-- Function to convert and clean up JSON
CREATE OR REPLACE FUNCTION public.convert_and_cleanup_json(input_text text)
RETURNS jsonb AS $$
DECLARE
    result jsonb := '{}'::jsonb;
    pairs text[];
    pair text;
    key text;
    value text;
BEGIN
    -- Check for null or empty input
    IF input_text IS NULL OR input_text = '{}' OR input_text = '' THEN
        RETURN NULL;
    END IF;

    -- Remove leading and trailing curly braces and split into pairs
    pairs := regexp_split_to_array(trim(both '{}' FROM input_text), '","');
    
    FOREACH pair IN ARRAY pairs
    LOOP
        -- Split each pair into key and value
        key := split_part(pair, ',', 1);
        value := split_part(pair, ',', 2);
        
        -- Clean up key and value
        key := regexp_replace(trim(both '"' FROM trim(both '(' FROM key)), '[()]', '', 'g');
        value := regexp_replace(trim(both '"' FROM trim(both ')' FROM value)), '[()]', '', 'g');
        
        -- Handle the case where there's no comma (it's all part of the key)
        IF value = '' THEN
            key := regexp_replace(trim(both '"' FROM trim(both '(' FROM pair)), '[()]', '', 'g');
            value := '';
        END IF;
        
        -- Trim any resulting leading/trailing spaces
        key := trim(both ' ' FROM key);
        value := trim(both ' ' FROM value);
        
        -- Add to result
        result := jsonb_set(result, ARRAY[key], to_jsonb(value));
    END LOOP;

    RETURN result;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Error in convert_and_cleanup_json: %', SQLERRM;
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;

