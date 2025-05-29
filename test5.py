from memory_profiler import profile

@profile
def my_func():
    for i in range(5):
        a = [1] * 1000000
        b = [2] * 9000000
    del b
    return a

my_func()



def test(device, trial):
    model.eval()
    model.to(device)  # モデルをGPUへ移動
    
    temp = 0
    ddd = 0

    for step, data in enumerate(test_data_loader):
        
        with torch.no_grad():  # 勾配を計算しない
            metric_loss = 0
            
            for i in range(len(data)):
                
                # polygonID取得
                polygonID = int(data[i].split("_")[-1])

                # polygonデータの取得と正規化 (すべてGPUへ移動)
                polygon = data_getter(polygonID, 0, test_mesh_data_lists[trial], test_polygon_data_list)
                polygon = normalization(polygon)
                polygon.coordinates = polygon.coordinates.to(device)

                # 入力データの準備
                x = polygon.coordinates.unsqueeze(0).to(device)
                out = model(x)  # モデルの出力はGPU上

                # ログを出力
                logger.info("epoch: {:04}, polygonID: {:04}".format(epoch, polygonID))
                logger.info("before")
                
                ml = criterion(polygon)  # GPU上の計算
                logger.info("")
                logger.info("after")

                # 損失関数の計算 (すべてGPU上)
                l = criterion(polygon, out[0])
                logger.info("")

                # メトリクスの更新
                metric_loss += l

                # 座標の更新 (GPU上)
                polygon.coordinates[0] += out[0]
                polygon = denormalization(polygon)
                
                # メッシュの更新
                polygon_meshID = int(test_polygon_data_list[polygonID].meshID.split("_")[-1])
                mesh = test_mesh_data_lists[trial][polygon_meshID]
                
                # GPUの値を取得し、メッシュデータに適用（GPU上で保持）
                mesh.coordinates[test_polygon_data_list[polygonID].nodeID[0]] = polygon.coordinates[0].cpu()

            # バッチの平均損失を計算
            loss = metric_loss / len(data)
            ddd += len(data)
            temp += loss
        
    # 最終的な損失の平均
    test_loss_ave = temp / ddd
    writer.add_scalar("loss", test_loss_ave, epoch)       
    print(test_loss_ave, epoch)
    loss_list.append(test_loss_ave)
