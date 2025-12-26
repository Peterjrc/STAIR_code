import os
import numpy as np
import pandas as pd

data_dir = './experiment_data/'
colors = np.array(['#377eb8', '#ff7f00', '#0000FF', '#FFFF00',
                   '#FFA500', '#FF0000', '#008000', '#808080',
                   '#800080', '#FFD700', "#B1BBB8", "#F3E6C3"])

############################################################################################################
# draw pictures on stair and stair with penalty term
lm_effect_df = pd.read_csv(os.path.join(data_dir, 'effect_lm_stair','effect_lm_stair.csv'))
lm_effect_df_penalty = pd.read_csv(os.path.join(data_dir, 'effect_lm_lstair','effect_lm_lstair.csv'))

import matplotlib.pyplot as plt

# # Data for the plot
# for name in lm_effect_df.columns[1:]:
#     max_length = [2, 4, 6, 8, 10, 12]
#     stair_lengths = lm_effect_df.loc[:, name]
#     stair_penalty_lengths = lm_effect_df_penalty.loc[:,name]
#     # Create the plot
#     plt.figure(figsize=(6, 5))
#
#     # Plot for STAIR
#     plt.plot(max_length, stair_lengths, marker='o', linestyle='-', color='darkblue', label='STAIR', linewidth=4, markersize=10)
#     plt.plot(max_length, stair_penalty_lengths, marker='s', linestyle='-', color='green', label='L-STAIR', linewidth=4, markersize=10)
#     # Plot for L-STAIR
#     # plt.plot(max_length, lstair_lengths, marker='s', linestyle='-', color='green', label='L-STAIR', linewidth=2)
#
#     # Add title and labels
#     # plt.title('Sum of Lengths vs Max Length')
#     plt.xlabel('Max Length', fontsize=16)
#     plt.ylabel('Sum of Lengths', fontsize=16)
#
#     # Add gridlines
#     plt.grid(True, linestyle='--', color='lightblue')
#     plt.tick_params(axis='both', which='major', labelsize=15)
#
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#
#     plt.xticks( fontsize=15, fontweight='bold')
#     plt.yticks(fontsize=15, fontweight='bold')
#     # Add legend
#     legend = plt.legend(prop={'size': 16, 'weight': 'bold'})
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(data_dir, 'effect_lm_lstair', f'{name}_max_length_lstair.pdf'), format='pdf')
#     # Show the plot
#     plt.show()

############################################################################################################

# 画stair penalty 对 recall score的影响
# recall_stair_df = pd.read_csv(os.path.join(data_dir,'others','recall_stair_penalty.csv'), index_col = 0)
# recall_stair_df = recall_stair_df.transpose()
# data_list = recall_stair_df.columns[:4]
# recall_score_penalty = recall_stair_df.iloc[0,:4]
# recall_score = recall_stair_df.iloc[1,:4]
#
# x = np.arange(len(data_list))  # 柱子的位置
# width = 0.35  # 柱子的宽度
#
# # 创建图形和坐标轴对象
# fig, ax = plt.subplots(figsize = (6,5))
#
# # 绘制两组柱状图
# rects2 = ax.bar(x - width/2, recall_score_penalty, width, label='Recall Score STAIR+', color = 'pink', hatch = 'o', edgecolor='black')
# rects1 = ax.bar(x + width/2, recall_score, width, label='Recall Score STAIR', color = 'lightblue', hatch = 'x', edgecolor='black')
#
# # 设置坐标轴标签和标题
# ax.set_ylabel('Recall Score', fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels(data_list, fontweight='bold')  # 旋转 x 轴标签以避免重叠
# ax.legend(loc='upper center', bbox_to_anchor=(0.2, 1.14), prop={'size': 10, 'weight': 'bold'})
# ax.set_ylim(0.6, 1)
#
# # 让上方和右侧边框消失
# # ax.spines['right'].set_visible(False)
# # ax.spines['top'].set_visible(False)
#
#
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontweight('bold')
#
# # 显示图表
# plt.tight_layout()  # 自动调整布局
# plt.savefig(os.path.join(data_dir, 'recall_stair_penalty.pdf'), format='pdf')
# plt.show()

############################################################################################################
# 画M的动态折线图stair

# for name in ["Annthyroid", "skin", "optdigits", "ALOI", 'Pendigits', 'Pima', "Mammography", "Satimage-2", "PageBlock", "satellite", 'Thursday']:
#     M_stair_df = pd.read_csv(os.path.join(data_dir,'M_value_stair',f'M_stair_{name}.csv'), index_col = 0)
#     plt.figure(figsize=(6, 5))
#
#     number_of_rules = M_stair_df.loc[:,'Rule_num']
#     M_value = M_stair_df.loc[:,'M_value']
#     # 绘制折线图
#     plt.plot(number_of_rules, M_value, marker='o', linestyle='-', color='#377EB8', linewidth=2, markersize=4, label = 'Normal')
#
#     # 设置标题和轴标签
#     plt.xlabel('Number of Rules', fontsize=19)
#     plt.ylabel('The Value of M', fontsize=19)
#
#     # 设置刻度
#     plt.tick_params(axis='both', which='major', labelsize=18)
#
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     # 添加网格线
#     plt.grid(True, linestyle='--', color = 'lightblue')
#
#     plt.xticks(np.arange(min(number_of_rules), max(number_of_rules) + 1, 30), fontsize=18, fontweight='bold')
#     plt.yticks(fontsize=18, fontweight='bold')
#     # Add legend
#     legend = plt.legend(prop={'size': 16, 'weight': 'bold'})
#
#     # 显示图形
#     plt.tight_layout()
#     plt.savefig(os.path.join(data_dir, 'M_value_stair', f'M_value_stair_{name}.pdf'), format='pdf')
#     plt.show()



############################################################################################################
# draw pictures on Q3 effect of lm on stair and lstair
# lm_effect_df = pd.read_csv(os.path.join(data_dir, 'effect_lm_stair','effect_lm_stair.csv'))
# lm_effect_df_penalty = pd.read_csv(os.path.join(data_dir, 'effect_lm_lstair','effect_lm_lstair.csv'))
#
# import matplotlib.pyplot as plt
#
# # Data for the plot
# for name in lm_effect_df.columns[1:]:
#     if name in ['PageBlock', 'Pendigits', 'Pima', 'Mammography', 'satellite', 'Thursday']:
#         max_length = [2, 4, 6, 8, 10, 12]
#         stair_lengths = lm_effect_df.loc[:, name]
#         stair_penalty_lengths = lm_effect_df_penalty.loc[:, name]
#     else:
#         max_length = [4, 6, 8, 10, 12]
#         stair_lengths = lm_effect_df.loc[1:, name]
#         stair_penalty_lengths = lm_effect_df_penalty.loc[1:,name]
#     # Create the plot
#     plt.figure(figsize=(6, 5))
#
#     # Plot for STAIR
#     plt.plot(max_length, stair_lengths, marker='o', linestyle='-', color='darkblue', label='STAIR', linewidth=4, markersize=10)
#     # plot for L-STAIR
#     plt.plot(max_length, stair_penalty_lengths, marker='o', linestyle='-', color='green', label='L-STAIR', linewidth=4, markersize=10)
#
#     # Add title and labels
#     # plt.title('Sum of Lengths vs Max Length')
#     plt.xlabel('Max Length', fontsize=19)
#     plt.ylabel('Sum of Lengths', fontsize=19)
#     # add legend
#     plt.legend(prop={'size': 16, 'weight': 'bold'})
#     # Add gridlines
#     plt.grid(True, linestyle='--', color='lightblue')
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#
#     plt.xticks(max_length, fontsize=18, fontweight='bold')
#     plt.yticks(fontsize=18, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(os.path.join(data_dir, 'effect_lm_lstair', f'{name}_max_length_lstair.pdf'), format='pdf')
#     # Show the plot
#     plt.show()
#
#     lm_effect_df = pd.read_csv(os.path.join(data_dir, 'effect_lm_stair', 'effect_lm_stair.csv'))
#     lm_effect_df_penalty = pd.read_csv(os.path.join(data_dir, 'effect_lm_lstair', 'effect_lm_lstair.csv'))

############################################################################################################

# draw pictures for lstair and lstair with penalty term
# import matplotlib.pyplot as plt
#
# lm_effect_df = pd.read_csv(os.path.join(data_dir, 'effect_lm_lstair','effect_lm_lstair.csv'))
# lm_effect_df_penalty = pd.read_csv(os.path.join(data_dir, 'effect_lm_lstair_penalty','effect_lm_lstair_penalty.csv'))
#     # Data for the plot
# for name in lm_effect_df.columns[1:]:
#     max_length = [2, 4, 6, 8, 10, 12]
#     stair_lengths = lm_effect_df.loc[:, name]
#     stair_penalty_lengths = lm_effect_df_penalty.loc[:, name]
#     # Create the plot
#     plt.figure(figsize=(8, 6))
#
#     # Plot for STAIR
#     plt.plot(max_length, stair_lengths, marker='o', linestyle='-', color='green', label='L-STAIR', linewidth=2)
#     plt.plot(max_length, stair_penalty_lengths, marker='o', linestyle='-', color='orange', label='L-STAIR+',
#              linewidth=2)
#     # Plot for L-STAIR
#     # plt.plot(max_length, lstair_lengths, marker='s', linestyle='-', color='green', label='L-STAIR', linewidth=2)
#
#     # Add title and labels
#     # plt.title('Sum of Lengths vs Max Length')
#     plt.xlabel('Max Length', fontsize=10)
#     plt.ylabel('Sum of Lengths', fontsize=10)
#
#     # Add gridlines
#     plt.grid(True, linestyle='--', color='lightblue')
#
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#
#     plt.xticks(fontsize=10, fontweight='bold')
#     plt.yticks(fontsize=10, fontweight='bold')
#     # Add legend
#     plt.legend()
#
#     plt.savefig(os.path.join(data_dir, 'effect_lm_lstair_penalty', f'{name}_max_length_lstair_penalty.pdf'), format='pdf')
#     # Show the plot
#     plt.show()
############################################################################################################

# 画每一个类别的早期训练损失
# early_loss_df = pd.read_csv(os.path.join(data_dir, 'others','early_loss_class_optdigits.csv'))
# print(early_loss_df)
# mislabel_inlier_df = early_loss_df[early_loss_df['class'] == 1]
# clean_inlier_df = early_loss_df[early_loss_df['class'] == 0]
# mislabel_outlier_df = early_loss_df[early_loss_df['class'] == -1]
# clean_outlier_df = early_loss_df[early_loss_df['class'] == 2]
#
# mislabel_df = pd.concat([mislabel_inlier_df, mislabel_outlier_df], axis=0)
# print(len(mislabel_df))
# clean_df = pd.concat([clean_inlier_df, clean_outlier_df], axis=0)
# # plt.hist(clean_outlier_df['loss'], bins=100, color='red', alpha=0.5, label='Mislabel Inlier')
# # plt.hist(mislabel_outlier_df['loss'], bins=100, color='blue', alpha=0.5, label='mislabel outlier')
# # bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# # labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# # clean_df['range'] = pd.cut(clean_df['loss'], bins=bins, labels=labels, right=False)
# # range_counts_clean = clean_df['range'].value_counts().sort_index()
# # print(range_counts_clean)
# #
# # bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# # labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# # mislabel_df['range'] = pd.cut(mislabel_df['loss'], bins=bins, labels=labels, right=False)
# # range_counts_mislabel = mislabel_df['range'].value_counts().sort_index()
# # print(range_counts_mislabel)
#
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# bins = [0.2, 0.4, 0.6, 0.8, 1.0]
# labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# labels = ['[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# mislabel_df['range'] = pd.cut(mislabel_df['loss'], bins=bins, labels=labels, right=False)
# mislabel_counts = mislabel_df['range'].value_counts().sort_index()
#
# # bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# # labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# # mislabel_inlier_df['range'] = pd.cut(mislabel_inlier_df['loss'], bins=bins, labels=labels, right=False)
# # range_counts_mislabel_inlier = mislabel_inlier_df['range'].value_counts().sort_index()
#
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# bins = [0.2, 0.4, 0.6, 0.8, 1.0]
# labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# labels = ['[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# clean_df['range'] = pd.cut(clean_df['loss'], bins=bins, labels=labels, right=False)
# clean_counts = clean_df['range'].value_counts().sort_index()
#
#
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# # bins = [0.2, 0.4, 0.6, 0.8, 1.0]
# labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# # labels = ['[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# mislabel_outlier_df['range'] = pd.cut(mislabel_outlier_df['loss'], bins=bins, labels=labels, right=False)
# mislabel_outlier_counts = mislabel_outlier_df['range'].value_counts().sort_index()
#
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# # bins = [0.2, 0.4, 0.6, 0.8, 1.0]
# labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# # labels = ['[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# mislabel_inlier_df['range'] = pd.cut(mislabel_inlier_df['loss'], bins=bins, labels=labels, right=False)
# mislabel_inlier_counts = mislabel_inlier_df['range'].value_counts().sort_index()
#
#
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# # bins = [0.2, 0.4, 0.6, 0.8, 1.0]
# labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# # labels = ['[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# clean_outlier_df['range'] = pd.cut(clean_outlier_df['loss'], bins=bins, labels=labels, right=False)
# clean_outlier_counts = clean_outlier_df['range'].value_counts().sort_index()
#
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# # bins = [0.2, 0.4, 0.6, 0.8, 1.0]
# labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# # labels = ['[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# clean_inlier_df['range'] = pd.cut(clean_inlier_df['loss'], bins=bins, labels=labels, right=False)
# clean_inlier_counts = clean_inlier_df['range'].value_counts().sort_index()
# # bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# # labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0)']
# # mislabel_outlier_df['range'] = pd.cut(mislabel_outlier_df['loss'], bins=bins, labels=labels, right=False)
# # range_counts_mislabel_outlier = mislabel_outlier_df['range'].value_counts().sort_index()
#
# mislabel_counts = mislabel_counts.values
# clean_counts = clean_counts.values
# clean_inlier_counts = clean_inlier_counts.values
# clean_outlier_counts = clean_outlier_counts.values
# mislabel_inlier_counts = mislabel_inlier_counts.values
# mislabel_outlier_counts = mislabel_outlier_counts.values
#
# plt.figure(figsize=(8, 6))
# plt.bar(labels, clean_inlier_counts, hatch = 'x' ,label='clean Instances', color = 'red')
# plt.bar(labels,mislabel_inlier_counts, bottom=clean_inlier_counts, hatch = 'o' ,label='mislabel Instances', color = 'green')
#
# # bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# # labels = ['[0, 0.1)', '[0.1, 0.3)', '[0.3, 0.5)', '[0.5, 0.7)', '[0.7, 1.0)']
# # mislabel_outlier_df['range'] = pd.cut(mislabel_outlier_df['loss'], bins=bins, labels=labels, right=False)
# # range_counts_mislabel_outlier = mislabel_outlier_df['range'].value_counts().sort_index()
# # print(range_counts_mislabel_outlier)
#
# plt.title('Early Loss Distribution')
# plt.xlabel('Normalized range loss')
# plt.ylabel('Number of instances')
# print(mislabel_inlier_df['loss'])
# # plt.hist(mislabel_inlier_df['loss'], bins=100, color='red', alpha=0.5, label='Mislabel Inlier')
# # plt.hist(clean_inlier_df['loss'], bins=100, color='blue', alpha=0.5, label='clean inlier')
# plt.hist(clean_outlier_df['loss'], bins=200, color='orange', alpha=0.5, label='clean outlier')
# # plt.hist(mislabel_outlier_df['loss'], bins=200, color='green', alpha=0.5, label='Mislabel outlier')
# # #
# #
# # #
# plt.legend()
# #
# plt.show()

#########################################################################################################
#画stair lstair Q4
# f1_effect_df_stair = pd.read_csv(os.path.join(data_dir, 'effect_f1_stair','effect_f1_stair.csv'), index_col = 0)
# f1_effect_df_lstair = pd.read_csv(os.path.join(data_dir, 'effect_f1_lstair','effect_f1_lstair.csv'), index_col = 0)
# f1_effect_df_id3 = pd.read_csv(os.path.join(data_dir, 'effect_f1_id3','effect_f1_id3.csv'), index_col = 0)
# f1_effect_df_cart = pd.read_csv(os.path.join(data_dir, 'effect_f1_cart','effect_f1_cart.csv'), index_col = 0)
# print(f1_effect_df_stair)
# import matplotlib.pyplot as plt
# # Data for the plot
# for name in f1_effect_df_stair.columns:
#     print(name)
#     if name in ['satellite', 'Annthyroid']:
#         f1_score = [0.7, 0.75, 0.8, 0.85, 0.9]
#         stair_lengths = f1_effect_df_stair.loc[:0.9, name]
#         lstair_lengths = f1_effect_df_lstair.loc[:0.9, name]
#         id3_lengths = f1_effect_df_id3.loc[:0.9, name]
#         cart_lengths = f1_effect_df_cart.loc[:0.9, name]
#     else:
#         f1_score = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#         stair_lengths = f1_effect_df_stair.loc[:, name]
#         lstair_lengths = f1_effect_df_lstair.loc[:,name]
#         id3_lengths = f1_effect_df_id3.loc[:,name]
#         cart_lengths = f1_effect_df_cart.loc[:, name]
#
#     # Create the plot
#     plt.figure(figsize=(6, 5))
#
#     # Plot for STAIR
#     plt.plot(f1_score, id3_lengths, marker='o', linestyle='-', color='darkblue', label='ID3', linewidth=4, markersize=10)
#     plt.plot(f1_score, cart_lengths, marker='v', linestyle='-', color='#87CEFA', label='CART', linewidth=4,
#              markersize=10)
#     plt.plot(f1_score, stair_lengths, marker='^', linestyle='-', color='darkorange', label='STAIR', linewidth=4, markersize=10)
#     plt.plot(f1_score, lstair_lengths, marker='s', linestyle='-', color='#32CD32', label='L-STAIR', linewidth=4, markersize=10)
#
#     # Plot for L-STAIR
#     # plt.plot(max_length, lstair_lengths, marker='s', linestyle='-', color='green', label='L-STAIR', linewidth=2)
#
#     # Add title and labels
#     # plt.title('Sum of Lengths vs Max Length')
#     # plt.title(f'{name}')
#     plt.xlabel('Threshold', fontsize=19)
#     plt.ylabel('Sum of Lengths', fontsize=19)
#
#     # Add gridlines
#     plt.grid(True, linestyle='--', color='lightblue')
#     # plt.tick_params(axis='both', which='major', labelsize=15)
#
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#
#     if name in ['satellite', 'Annthyroid']:
#         xticks =[0.7, 0.75, 0.8, 0.85, 0.9]
#     else:
#         xticks = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#     plt.xticks(xticks, fontsize=18, fontweight='bold')
#     plt.yticks(fontsize=18, fontweight='bold')
#     # Add legend
#     legend = plt.legend(prop={'size': 16, 'weight': 'bold'})
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(data_dir, 'effect_f1_lstair', f'{name}_f1_lstair.pdf'), format='pdf')
#     # Show the plot
#     plt.show()

##################################################################
# 画influence function 的消融实验
# influence_evaluation_df = pd.read_csv(os.path.join(data_dir,'others', 'influence_ablation.csv'), index_col = 0)
# data_list = influence_evaluation_df.index
# with_influence = influence_evaluation_df.iloc[:,0]
# without_influence = influence_evaluation_df.iloc[:,1]
# print(with_influence)
# x = np.arange(len(data_list))  # 柱子的位置
# width = 0.35  # 柱子的宽度
#
# # 创建图形和坐标轴对象
# fig, ax = plt.subplots(figsize = (6,5))
#
# # 绘制两组柱状图
# rects1 = ax.bar(x - width/2, with_influence, width, label='With Influence', color = 'pink', hatch = '//', edgecolor='black')
# rects2 = ax.bar(x + width/2, without_influence, width, label='Without Influence', color = 'lightgreen', hatch = '\\\\', edgecolor='black')
#
# # 设置坐标轴标签和标题
# ax.set_ylabel('F1 score', fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels(data_list, fontweight='bold')  # 旋转 x 轴标签以避免重叠
# ax.legend(loc='upper center', bbox_to_anchor=(0.19, 1.14), prop={'size': 10, 'weight': 'bold'})
# ax.set_ylim(0.4, 0.8)
#
# # 让上方和右侧边框消失
# # ax.spines['right'].set_visible(False)
# # ax.spines['top'].set_visible(False)
#
#
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontweight('bold')
#
# # 显示图表
# plt.tight_layout()  # 自动调整布局
# plt.savefig(os.path.join(data_dir, 'influence_evaluation.pdf'), format='pdf')
# plt.show()
#
# ###################################################################################################
# #画early stop 消融实验
early_stop_df = pd.read_csv(os.path.join(data_dir,'others', 'early_stop_ablation.csv'), index_col = 0)
data_list = early_stop_df.index
with_early_stop = early_stop_df.iloc[:,0]
without_early_stop = early_stop_df.iloc[:,1]
x = np.arange(len(data_list))  # 柱子的位置
width = 0.35  # 柱子的宽度

# 创建图形和坐标轴对象
fig, ax = plt.subplots(figsize = (6,5))

# 绘制两组柱状图
rects1 = ax.bar(x - width/2, with_early_stop, width, label='With Early Stop', color = 'orange', hatch = '//', edgecolor='black')
rects2 = ax.bar(x + width/2, without_early_stop, width, label='Without Early Stop', color = 'lightblue', hatch = '\\\\', edgecolor='black')

# 设置坐标轴标签和标题
ax.set_ylabel('F1 score', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(data_list, fontweight='bold')  # 旋转 x 轴标签以避免重叠
ax.legend(loc='upper center', bbox_to_anchor=(0.19, 1.14), prop={'size': 10, 'weight': 'bold'})
ax.set_ylim(0.35, 0.7)

# 让上方和右侧边框消失
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)


for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

# 显示图表
plt.tight_layout()  # 自动调整布局
plt.savefig(os.path.join(data_dir, 'early_stop_ablation.pdf'), format='pdf')
plt.show()


##################################################################
#画 cbs 消融实验
# cbs_df = pd.read_csv(os.path.join(data_dir,'others', 'cbs_ablation.csv'), index_col = 0)
# data_list = cbs_df.index
# with_cbs = cbs_df.iloc[:,0] * 100
# without_cbs = cbs_df.iloc[:,1] * 100
# x = np.arange(len(data_list))  # 柱子的位置
# width = 0.35  # 柱子的宽度
#
# # 创建图形和坐标轴对象
# fig, ax = plt.subplots(figsize = (6,5))
#
# # 绘制两组柱状图
# rects1 = ax.bar(x - width/2, with_cbs, width, label='With Class Balance Sampling', color = 'purple', hatch = '//', edgecolor='black')
# rects2 = ax.bar(x + width/2, without_cbs, width, label='Without Class Balance Sampling', color = 'lightgrey', hatch = '\\\\', edgecolor='black')
#
# # 设置坐标轴标签和标题
# ax.set_ylabel('Detected Mislabeled Inliers Percentage (%)', fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels(data_list, fontweight='bold')  # 旋转 x 轴标签以避免重叠
# ax.legend(loc='upper center', bbox_to_anchor=(0.3, 1.14), prop={'size': 10, 'weight': 'bold'})
# ax.set_ylim(0, 60)
#
# # 让上方和右侧边框消失
# # ax.spines['right'].set_visible(False)
# # ax.spines['top'].set_visible(False)
#
#
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontweight('bold')
#
# # 显示图表
# plt.tight_layout()  # 自动调整布局
# plt.savefig(os.path.join(data_dir, 'cbs_ablation.pdf'), format='pdf')
# plt.show()