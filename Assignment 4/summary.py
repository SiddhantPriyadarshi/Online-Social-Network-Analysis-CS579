import pickle

def main():
    summary = open('summary.txt','w',encoding = 'utf-8')
    collect = open('collect.txt','r')
    cluster = open('cluster.txt','r')
    classify = open('classify.txt','r')
    summary.write(collect.read())
    summary.write('\n')
    summary.write(cluster.read())
    summary.write('\n')
    summary.write(classify.read())
    summary.write('\n')
    summary.close()
    collect.close()
    cluster.close()
    classify.close()

if __name__ == "__main__":
    main()