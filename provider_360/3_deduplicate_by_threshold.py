# Databricks notebook source
# MAGIC %md
# MAGIC ## Purpose
# MAGIC
# MAGIC This notebook filters potential duplicate records based on entity-specific similarity score thresholds, creating a "gold" table.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC *   You must run notebook 1, as that model creates the correct Vector Search DB and the temporary candidate views.
# MAGIC *   You must run notebook 2, as that creates the data set used for model parameterization and the definition of appropriate thresholds.
# MAGIC *   Make sure the threshold parameters and the bronze table is correctly specified.
# MAGIC
# MAGIC ## Outputs
# MAGIC
# MAGIC *   Writes the filtered data for each entity to its corresponding gold table. After running this, you should see a new table with a specific model that is deduped.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### dbutils Parameters
# MAGIC The gold table is the only thing we need a dbutils parameter for here, but you will have to verify the table parameters such as catalog, database, name.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "", "Catalog Name")
catalog_name = dbutils.widgets.get("catalog_name")
if not catalog_name:
    raise Exception("Catalog name is required to run this notebook")
gold_table_root = f"{catalog_name}.gold"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Parameters
# MAGIC You may have a number of entities to process with a certain degree of complexity. Here, you will want to set table names, model keys, and then tweak individual parameters so the model has the most accurate results. After a model runs, you can adjust these params.
# MAGIC
# MAGIC These params can be found in Notebook 1 and 2.

# COMMAND ----------

# Define the entities to process (same as Notebook 1)
entities = [
    {"table_name": f"{catalog_name}.bronze.provider", "primary_key": "provider_id", "columns_to_exclude": ["provider_id"]},
    {"table_name": f"{catalog_name}.bronze.speciality", "primary_key": "speciality_id", "columns_to_exclude": ["speciality_id", "provider_id"]},
    {"table_name": f"{catalog_name}.bronze.license_and_credential", "primary_key": "license_credential_id", "columns_to_exclude": ["license_credential_id", "provider_id"]},
    {"table_name": f"{catalog_name}.bronze.affiliations", "primary_key": "affiliation_id", "columns_to_exclude": ["affiliation_id", "provider_id"]},
    {"table_name": f"{catalog_name}.bronze.address_and_location", "primary_key": "address_location_id", "columns_to_exclude": ["address_location_id", "provider_id"]},
    {"table_name": f"{catalog_name}.bronze.contact_information", "primary_key": "contact_id", "columns_to_exclude": ["contact_id", "provider_id"]},
    {"table_name": f"{catalog_name}.bronze.ProviderNetworkParticipation", "primary_key": "network_id", "columns_to_exclude": ["network_id", "provider_id"]},
    {"table_name": f"{catalog_name}.bronze.Employment_and_Contracts", "primary_key": "employment_contract_id", "columns_to_exclude": ["employment_contract_id", "provider_id"]},
    {"table_name": f"{catalog_name}.bronze.Education_and_Training", "primary_key": "education_training_id", "columns_to_exclude": ["education_training_id", "provider_id"]},
    {"table_name": f"{catalog_name}.bronze.Performance_and_QualityMetrics", "primary_key": "performance_metric_id", "columns_to_exclude": ["performance_metric_id", "provider_id"]},
    {"table_name": f"{catalog_name}.bronze.Identifiers", "primary_key": "identifier_id", "columns_to_exclude": ["identifier_id", "provider_id"]},
    {"table_name": f"{catalog_name}.bronze.Digital_Presence", "primary_key": "digital_presence_id", "columns_to_exclude": ["digital_presence_id", "provider_id"]}
]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Parameters
# MAGIC You must properly specify these thresholds based on the recommendations output by Notebook 2. If it recommends .95 for provider, then make sure you have that. The gold table in this case refers to the table being built *after* the model is completed and may overwrite the bronze table, based on configuration.

# COMMAND ----------

# Define the thresholds for each entity
thresholds = {
    f"{catalog_name}.bronze.provider": 0.95,
    f"{catalog_name}.bronze.speciality": 0.90,
    f"{catalog_name}.bronze.license_and_credential": 0.85,
    f"{catalog_name}.bronze.affiliations": 0.92,
    f"{catalog_name}.bronze.address_and_location": 0.88,
    f"{catalog_name}.bronze.contact_information": 0.91,
    f"{catalog_name}.bronze.ProviderNetworkParticipation": 0.93,
    f"{catalog_name}.bronze.Employment_and_Contracts": 0.87,
    f"{catalog_name}.bronze.Education_and_Training": 0.89,
    f"{catalog_name}.bronze.Performance_and_QualityMetrics": 0.94,
    f"{catalog_name}.bronze.Identifiers": 0.96,
    f"{catalog_name}.bronze.Digital_Presence": 0.97
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Process the entity
# MAGIC In this step we are applying the model to our dataset. You must verify your tables have the correct names or this will fail.

# COMMAND ----------

# Process each entity
for entity in entities:
    table_name = entity["table_name"]
    primary_key = entity["primary_key"]
    print(primary_key)
    entity_name = table_name.split(".")[-1] # Extract entity name from table_name
    candidate_view = f"{table_name}_duplicate_candidates" # Construct the name of the duplicate candidate view
    gold_table = f"{gold_table_root}.{entity_name}" # Construct full gold table name

    # Get threshold for the current table
    if table_name in thresholds:
        threshold = thresholds[table_name]
        print(f"Processing {table_name} with threshold {threshold}")
    else:
        print(f"No threshold found for table {table_name} in the thresholds dictionary. Skipping...")
        continue # Skip to the next entity

    # Table Existence Check
    try:
        bronze_df = spark.table(table_name)
    except Exception as e:
        print(f"Error: Table '{table_name}' not found. Please check the table name.")
        continue

    # Check if the duplicate candidate view exists
    try:
        spark.table(candidate_view)
    except Exception as e:
        print(f"Error: Duplicate candidate view '{candidate_view}' not found. Please check the view name.")
        continue

    # Filter duplicates
    filtered_data = spark.sql(f"""
        SELECT t1.*
        FROM {table_name} t1
        LEFT JOIN {candidate_view} t2
        ON t1.{primary_key} = t2.original_id
        WHERE t2.original_id IS NULL OR t2.search_score < {threshold}
    """)

    # Write filtered data to gold table
    filtered_data.write.mode("overwrite").saveAsTable(gold_table)
    spark.sql(f"DROP TABLE IF EXISTS {candidate_view}")
