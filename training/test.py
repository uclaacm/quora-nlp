import torch
from tqdm import tqdm, tqdm_notebook
import numpy as np
from sklearn.metrics import roc_auc_score

def get_correct(outputs, labels):
    n_sincere = sum([x == 0 for x in labels]).item()
    n_insincere = sum([x == 1 for x in labels]).item()
    
    n_correct_sincere = 0
    n_correct_insincere = 0
    
    for o, l in zip(outputs, labels):
        # print(round(torch.sigmoid(o).item()), l.item())
        if ((round(torch.sigmoid(o).item())) == l.item()):
            if l.item() == 0:
                n_correct_sincere += 1
            else:
                n_correct_insincere += 1
    
    return np.array([n_correct_sincere, n_sincere, n_correct_insincere, n_insincere])

def test_loop(model, test_loader, device, loss_criterion):
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm_notebook(test_loader, leave=False)
        losses = []
        stats = np.array([0,0,0,0])
        
        target_cumulative = np.array([])
        prediction_cumulative = np.array([])

        total = 0
        for inputs, target in progress_bar:
            inputs, target = inputs.to(device), target.to(device)
            model.zero_grad()

            output = model(inputs)
            loss = loss_criterion(output.squeeze(), target.float())

            progress_bar.set_description(f'Loss: {loss.item():.3f}')

            stats += get_correct(output, target)
            losses.append(loss.item())
            total += 1
            target_np = target.cpu().detach().numpy()
            target_cumulative = np.append(target_cumulative, target_np)
            prediction_np = (output.cpu().detach().numpy() > 0.5).astype(int)
            prediction_cumulative = np.append(prediction_cumulative, prediction_np)
            
        sincere_accuracy = stats[0] / stats[1] * 100
        insincere_accuracy = stats[2] / stats[3] * 100
        test_loss = sum(losses) / total
        
        tqdm.write(f'AUC Score: {roc_auc_score(target_cumulative, prediction_cumulative)}')
        tqdm.write(f'''Test Loss: {test_loss:.3f} \n
                    Sincere Accuracy: {stats[0]} / {stats[1]} ({sincere_accuracy:.3f}%) \n
                    Insincere Accuracy: {stats[2]} / {stats[3]} ({insincere_accuracy:.3f}%)''')
        