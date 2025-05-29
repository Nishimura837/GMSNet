# memory_profilerとline_profilerをインポート
from memory_profiler import profile
import torch

# メモリ使用量を計測したい関数にデコレーターを追加
@profile
def train(device):
    model.train()
    for step, data in enumerate(data_loader):
        empty_cache()
        gc.collect()
        # print(f"Step {step + 1}:")
        # print("==========")
        # print(data)
        # print(len(data))
        minibatch = Minibatch()
        minibatch_coordinates = []
        all_edge_index_1 = []
        all_edge_index_2 = []
        batch_list = []
        num_dis = 0
        metric_loss_list = []
        for i in range(len(data)):
            # gc.collect()
            polygonID = int(data[i].split("_")[-1])
            # print("polygonID:",polygonID)
            polygon = data_getter(polygonID, 0)
            # polygon を正規化する
            polygon = normalization(polygon)
            # print("polygon.coordinates:",polygon.coordinates)
            edge_index = polygon.edge_index + num_dis
            # print("polygon.edge_index:", edge_index)
            all_edge_index_1.append(edge_index[0])
            all_edge_index_2.append(edge_index[1])
            num_dis = num_dis + len(polygon.coordinates)
            # print("==========")
            minibatch_coordinates.append(polygon.coordinates)
            batch_i = torch.tensor([i]*len(polygon.coordinates))
            batch_list.append(batch_i)
            
        # print(minibatch_coordinates)
        minibatch.x = torch.cat(minibatch_coordinates, dim=0)
        
        # print(minibatch.x)
        # print("minibatch.x.size:", minibatch.x.size())
        edge_index_1 = torch.cat(all_edge_index_1, dim=-1)
        edge_index_2 = torch.cat(all_edge_index_2, dim=-1)
        minibatch.edge_index = torch.cat([edge_index_1.unsqueeze(0), edge_index_2.unsqueeze(0)], dim=0)
        # print(minibatch.edge_index)
        # print("minibatch.edge_index.size:", minibatch.edge_index.size())
        minibatch.batch = torch.cat(batch_list, dim=0)
        # print(minibatch.batch)
        # print("minibatch.batch.size:", minibatch.batch.size())

        # すべてのデータをGPU上に移動する
        minibatch.x = minibatch.x.to(device)
        minibatch.edge_index = minibatch.edge_index.to(device)
        minibatch.batch = minibatch.batch.to(device)
        model.to(device)

        out = model(minibatch.x, minibatch.edge_index, minibatch.batch)
        # print("out:", out)
        # print("out:", out.size())

        for i in range(len(data)):
            # gc.collect()
            polygonID = int(data[i].split("_")[-1])
            polygon = data_getter(polygonID, 0)
            # 正規化する
            polygon = normalization(polygon)

            polygon.coordinates[0] = out[i]
            # print("out_i:", out[i])
            polygon = check(polygon, polygonID)
            # print("fixed_out_i:", polygon.coordinates[0])
            metric_loss = criterion(polygon)
            metric_loss_list.append(metric_loss)
            

            # 非正規化する
            polygon = denormalization(polygon)

            # 予測したノードの座標をもとのメッシュに当てはめて更新する
            polygon_meshID = int(polygon_data_list[polygonID].meshID.split("_")[-1])
            mesh = mesh_data_list[polygon_meshID]

            mesh.coordinates[polygon_data_list[polygonID].nodeID[0]] = polygon.coordinates[0]
        
        

        loss = ((sum(metric_loss_list) / len(metric_loss_list)).requires_grad_(True))
        writer.add_scalar("Loss/train", loss, epoch)
        # print("loss:", loss)

        loss.backward()
        
        loss.detach()               # 計算グラフを切り離し、不要な計算グラフが保持されることを防ぐ

        optimizer.step()
        optimizer.zero_grad()

        # ステップごとに損失をログに記録
        writer.add_scalar("/mnt/logs", loss.item(), global_step=len(data_loader)*epoch + step)

        # いらない変数を削除
        del minibatch_coordinates, all_edge_index_1, all_edge_index_2, batch_list, out, num_dis, metric_loss_list, loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    train(device)