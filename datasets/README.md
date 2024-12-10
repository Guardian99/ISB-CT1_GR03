
- The workflow is designed to ensure the dataset adheres to a predefined schema, performing cleaning and validation before proceeding with further processing.Successfully validated and cleaned datasets are stored in Parquet format to ensure efficiency in storage and subsequent processing tasks(employees_cleaned.parquet) 
- We we split the dataset into three distinct parts: training (60%), testing (20%), and production (20%).  
- To ensure consistency and reproducibility of results, we used a fixed random seed during the splitting process. 
- Later, the segmented datasets were saved in Parquet file format for efficient storage and seamless access during model development and evaluation.(train,test and prod parquet files)