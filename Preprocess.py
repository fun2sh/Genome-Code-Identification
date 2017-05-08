import os
#READ GFF3 and get the positions of genes and nongenes

fasta= open("testsequence.gff3", 'U')
start= []
end = []
counter = 0
for line in fasta:
    line = line.strip().split("\t")
    if(counter == 2):
        name = line[0]
    if len(line) > 3 and counter%2==1:
        start.append(line[3])
        end.append(line[4])
    counter += 1

noncodingstart = []
noncodingend = []
noncodingstart.append(start[0])
noncodingend.append(start[1])

for a in range(1,len(start)-1):
    if(int(end[a]) < int(start[a+1])):
        noncodingstart.append(end[a])
        noncodingend.append(start[a+1])

noncodingstart.append(end[len(end)-1])
noncodingend.append(end[0])


fo = open("testnongene.bed",'w')
for a in range(len(noncodingstart)):
    fo.write(name + '\t' + noncodingstart[a] + '\t' + noncodingend[a] + '\n')
fo.close()
fo = open("testgene.bed",'w')
for a in range(1,len(start)):
    fo.write(name + '\t' + start[a] + '\t' + end[a] + '\n')
fo.close()

#Turn bed files into fasta files
os.system('samtools faidx testsequence.fasta')
os.system('bedtools getfasta -fi testsequence.fasta -bed testgene.bed -fo testgenes.fasta')
os.system('bedtools getfasta -fi testsequence.fasta -bed testnongene.bed -fo testnongenes.fasta')


#Genes
fasta= open("testgenes.fasta", 'r')
seq_length = []
gc_content = []
molecular_weight = []
nucleotides = {'A' : 313.21,'G':329.21,'C':289.18,'T':304.2}
GC = 0.0
for line in fasta:
    MW = 0.0
    line= line.strip()
    if line == '':
        continue
    if line.startswith('>'):
        seqname= line.lstrip('>')
    else:
        #FEATURES
        seq_length.append(len(line))
        GC = line.count('C') + line.count('G')
        gc_content.append(GC/float(len(line)))
        for ch in line:
            MW += nucleotides[ch]
        molecular_weight.append(MW)
fasta.close()
noGenes = len(seq_length)

#Nongenes
fasta= open("testnongenes.fasta", 'r')
nucleotides = {'A' : 313.21,'G':329.21,'C':289.18,'T':304.2}
GC = 0.0
for line in fasta:
    MW = 0.0
    line= line.strip()
    if line == '':
        continue
    if line.startswith('>'):
        seqname= line.lstrip('>')
        #fasta_dict[seqname]= ''
    else:
        #FEATURES
        seq_length.append(len(line))
        GC = line.count('C') + line.count('G')
        gc_content.append(GC/float(len(line)))
        for ch in line:
            MW += nucleotides[ch]
        molecular_weight.append(MW)
fasta.close()

notGenes = len(seq_length) - noGenes


train= open('Test.txt', 'w')
for a in range(len(seq_length)):
    train.write(str(seq_length[a]) + '\t' + str(gc_content[a]) + '\t' + str(molecular_weight[a]) + '\n')
train.close()

labels = open('TestLabels.txt','w')
for a in range(noGenes):
    labels.write('1\t0' + '\n')
for a in range(notGenes):
    labels.write('0\t1' + '\n')
labels.close()