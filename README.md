# MaskGCT-ComfyUI
a custom node for [MaskGCT](https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct/README.md) to Zero-Shot Text-to-Speech

## Weights
weights will download from hf automatically,对于国内用户，你可以手动下载后，解压到ComfyUI/models/AIFSH文件夹，[下载地址MaskGCT](https://pan.quark.cn/s/f1b5a38e16dc),[下载地址w2v-bert-2.0.zip](https://pan.quark.cn/s/534d39e78610),[下载地址whisper-large-v3-turbo](https://pan.quark.cn/s/27ee36b1046b)

## Debug for espeak-ng on Windows
- [使用开箱即用的整合包](https://b23.tv/e6gQ9ms)
  - [使用演示](https://www.bilibili.com/video/BV1JSSxYuEfu)
  - 1键包1次充电31天免费更新，内含Ultralight-Digital-Human，GLM-4-Voice,MaskGCT，F5-TTS，FireRedTTS，hallo2，JoyHallo，SeedVC八个AIFSH的节点，持续更新中
- 我花费了1天时间debug， debug时间仅供参考，或许你更厉害呢，试试吧
## Disclaimer / 免责声明
We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws. 我们不对代码库的任何非法使用承担任何责任. 请参阅您当地关于 DMCA (数字千年法案) 和其他相关法律法规.
## Example
|target_text|prompt_wav|result_wav|
|--|--|--|
|`《三体》是刘慈欣创作的长篇科幻小说系列，由《三体》《三体2：黑暗森林》《三体3：死神永生》组成，第一部于2006年5月起在《科幻世界》杂志上连载，第二部于2008年5月首次出版，第三部则于2010年11月出版。`|<video src="https://github.com/user-attachments/assets/a237ab33-f51a-4ce0-9036-f8b29161c40d"/>|<video src="https://github.com/user-attachments/assets/1cd151e5-b913-4d81-85f5-c729a1f80e02"/>|
|`You cannot improve your past,but you can improve your future.Once time is wasted,life is wasted.`|<video src=""/>|<video src="https://github.com/user-attachments/assets/8b2203dc-0736-4206-b539-7d86d25e66f0"/>|
|`あなたの元（もと）へと帰りたい、背負（せお）った荷物置いて。あなたの元で眠（ねむ）りたい、あの頃のように穏（おだ）やかに。`|<video src=""/>|<video src="https://github.com/user-attachments/assets/57cbedd8-0964-463e-8f90-bd169c9d081a"/>|
|`나는 내 인생에서 실패에 실패를 거듭했다. 그런데 그것이 바로 내가 성공하는 이유이다. `|<video src=""/>|<video src="https://github.com/user-attachments/assets/a8957a55-7d78-4ec5-9d4b-2bc4102529d3"/>|
|`Prouver que j'ai raison serait accorder que je puis avoir tort.`|<video src=""/>|<video src="https://github.com/user-attachments/assets/1d25ef19-9c98-4596-a528-73f0da5e69ae"/>|
|`Egal wo du bist, wenn du in den Himmel schaust, schauen wir immer in den selben.`|<video src=""/>|<video src="https://github.com/user-attachments/assets/bef17e47-7ab0-4650-a536-035d6c237130"/>|
