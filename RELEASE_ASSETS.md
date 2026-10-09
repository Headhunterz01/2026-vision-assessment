# GitHub Release 媒体清单

这些视频保留在本机对应目录，根 `.gitignore` 将它们排除在 Git 提交之外。准备 Release 时，把视频作为附件上传，并保持下表中的文件名；下载后放回本机路径即可复现。

## 视频附件

| 用途 | 附件名 | 本机路径 | 大小 (MiB) | SHA-256 |
| --- | --- | --- | ---: | --- |
| 原题视频 | `armors_yolo_test.mp4` | `00_题目原件/videos/armors_yolo_test.mp4` | 93.20 | `0d139c1f86251071f41c9b60f3bf01c6109bb78b51ef3d5073158761e8098db0` |
| 原题视频 | `test.mp4` | `00_题目原件/videos/test.mp4` | 12.47 | `e1e9493e70c764ae2e170295e4a46c5c1087fd97d3d90118fe3e19d29aa68f74` |
| EKF 最终标注回放 | `ekf_tracking_final.mp4` | `04_EKF/results/ekf_tracking_final.mp4` | 55.69 | `a8e07f62e61420fc815e83738de9eadc9a19f4c04b57d952681e31686a650c1f` |
| 坐标变换最终回放 | `world_yaw.mp4` | `05_坐标变换/results/world_yaw.mp4` | 58.46 | `d72710d5392cf9c623de69358a68b4383be345a82079f171a1316b83e33d59ae` |
| 四装甲板重投影最终回放 | `four_plate_reprojection.mp4` | `06_重投影/results/four_plate_reprojection.mp4` | 79.79 | `9ccc7891a03ab8427c519e05b81a03a6142ab9a881865dd4729885899a132b95` |

当前目录尚未配置 GitHub remote，因此此清单记录附件与校验信息；发布时再将各考核 README 中的附件名称链接到实际 Release 下载 URL。
