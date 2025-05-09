import torch

# https://nn.labml.ai/diffusion/ddpm/utils.html
def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
    const = consts.gather(-1, t)
    return const.reshape(-1, 1, 1, 1)

# 예제 텐서
t = torch.tensor([[0, 1, 2], 
                  [3, 4, 5], 
                  [6, 7, 8]])

# 정수형 인덱스 텐서
index = torch.tensor([[0, 2, 1], 
                      [1, 0, 2], 
                      [2, 1, 0]])

# gather 사용
t_const = t.gather(-1, index)
t_const = t_const.reshape(-1, 1)
print(t_const)
