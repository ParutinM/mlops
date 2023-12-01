import torch
import torch.nn.functional as F
import wandb
from torchmetrics.classification import MulticlassAccuracy


def train_model(
    model, device, train_loader, optimizer, epoch, log_interval, dry_run
):
    accuracy = MulticlassAccuracy(num_classes=10)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{:05d}/{} ({:.0f}%)] \t Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                ),
                end="\r",
            )
            wandb.log(
                {
                    "accuracy": accuracy(
                        output.cpu(), target.cpu()
                    ).item(),
                    "loss": loss.item(),
                    "epoch": epoch,
                }
            )
            if dry_run:
                break
    wandb.log(
        {
            "conf_matrix": wandb.plot.confusion_matrix(
                probs=output.cpu().detach().numpy(),
                y_true=target.cpu().detach().numpy(),
                class_names=list(range(10)),
            )
        }
    )


def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return pred
