import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct_mask = pred.eq(target.view_as(pred))
        num_correct = correct_mask.sum().item()
    return num_correct/len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            print(pred[:, i] == target)
            correct = correct + torch.sum(pred[:, i] == target).item()
    return correct/len(target)