
Esse dataset é resultado da coleta de informações relativas ao uso de uma ferramenta de gestão de arquivos para empresas.
As informações são de um unico usuário.

O arquivo contem nove colunas:

n_acesso: (Numérico,discreto,quantitativa discreta.) /Se refere aos quantidade vezes que o usuário acessou um determinado arquivo.(é a quantidade total histórica.)

ddat_ano:(Numérico,discreto,qualitativa ordinal) /Se refere ao ano em que o usuário acessou o arquivo.

ddat_mes:(Numérico,discreto,qualitativa ordinal) /Se refere ao mês em que o usuário acessou o arquivo.

ddata_dia:(Numérico,discreto,qualitativa ordinal) /Se refere ao dia do mês em que o usuário acessou o arquivo.

ddat_dia_semana(Numérico,discreto,qualitativa ordinal) /Se refere ao dia da semana em que usuário acessou o arquivo.

ddat_semestre:(Numérico,discreto,qualitativa ordinal) /Se refere ao semestre do ano em que o usuário acessou o arquivo.

ddat_trimestre:(Numérico,discreto,qualitativa ordinal) /Se refer ao tremestre do ano em que o usuário acessou o arquivo.

darq_id:(Numérico,discreto,qualitativa nominal) /Se refere ao arquivo que o usuário de fato acessou naquele dia.

acess_id:(Numérico,discreto,Qualitativa nominal) /#Variavel target. Se refere ao arquivo que usuário acessará no dia seguinte.

O Dataset não possui valores faltantes.

É um problema de classificação com 342 classes diferentes, sendo que algumas só são registradas uma unica ocorrencia.
