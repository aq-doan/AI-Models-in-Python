def add_lag_and_split_data_temp():
  
  path = os.getcwd()
  print(path)
  dataset = pd.read_csv('/content/drive/My Drive/Dataset/RasPi/sensor_raw.csv',index_col=[0])
  dataset= dataset.reset_index()
  dataset = dataset[['GyroX', 'GyroY', 'GyroZ', 'AccX','AccY','AccZ','Target(Class)']]
  cols = [col for col in dataset.columns if col != 'Target(Class)']

  
  for i, target_cls in enumerate(dataset['Target(Class)'].unique()):
    new_dataset = dataset[dataset['Target(Class)'] == target_cls].copy()

  
    trn_size = int(0.6 * new_dataset.shape[0])
    val_size = int(0.8 * new_dataset.shape[0])
  
    new_dataset_trn = new_dataset[:trn_size].copy()
    new_dataset_val = new_dataset[trn_size:val_size].copy()
    new_dataset_tst = new_dataset[val_size:].copy()


  
    if i == 0:
      data_trn = new_dataset_trn
      data_val = new_dataset_val
      data_tst = new_dataset_tst
      continue

  
    data_trn = data_trn.append(new_dataset_trn)
    data_val = data_val.append(new_dataset_val)
    data_tst = data_tst.append(new_dataset_tst)

  
  x_trn = data_trn.drop(['Target(Class)'], axis=1)
  x_val = data_val.drop(['Target(Class)'], axis=1)
  x_tst = data_tst.drop(['Target(Class)'], axis=1)

  y_trn = data_trn['Target(Class)']
  y_val = data_val['Target(Class)']
  y_tst = data_tst['Target(Class)']

  return x_trn, x_val, x_tst, y_trn, y_val, y_tst
