import matplotlib.pyplot as plt
import pandas as pd

def time_convert(x):
    m,s = map(int,x.split(':'))
    return m*60+s

def plot(plot_name, grp, x, y='test accuracy'):
    plt.figure()
    for name, group in grp:
        if type(name) != str:
            label = '_'.join([str(x) for x in name])
        else:
            label = name
        plt.plot(group[x], group[y], label=label)
    plt.legend()
    plt.title(f"Test accuracy vs mode for augemented datasets")
    plt.savefig(f'new_plots/{plot_name}_{x}_vs_{y}.png')
    plt.close()

df = pd.read_csv('data.csv', sep=',', header=0)

info = df['file'].apply(lambda x: x.split('_'))
print(info)

df['lora_k'] = info.apply(lambda x: x[-1])
df['lora_type'] = info.apply(lambda x: int(x[-2]) if x[-2].isdigit() else {'same': -1, 'none': -2}.get(x[-2]))
df['num_layer'] = info.apply(lambda x: x[-3])
df['mode'] = info.apply(lambda x: int(x[-4]) if x[-4] != 'all' else -1)
# df['k'] = info.apply(lambda x: x[-5])
df['dataset'] = info.apply(lambda x: '_'.join(x[1:-5]))
df.drop(['file'], axis=1, inplace=True)

df['time done'] = df['time done'].apply(time_convert)
df['time per iter'] = df['time done']/df['iter']

df.to_csv('parsed_data.csv', sep=',', header=True, index=False)

df = df[df['dataset'] != 'cola'] # handle cola separately
# plot test accuracy vs mode for all grouped df
# lora mode, lora k, num layer
grp = df.groupby(['lora_type', 'lora_k', 'num_layer'])    
for n, g in grp:
    g2 = g.groupby('dataset')
    plot(f'lora_{n[0]}_{n[1]}_{n[2]}', g2, 'mode')