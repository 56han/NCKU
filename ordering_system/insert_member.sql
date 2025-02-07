INSERT INTO Member (MemName, Birth, Sex, JoinDate, Email, MemPassword, Points, Staff)
VALUES 
    (N'李小美', DATEADD(DAY, 2574, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '4667541@example.com', 'password1', 361, 'N'),
    (N'王明明', DATEADD(DAY, 2760, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '7923262@example.com', 'password2', 529, 'N'),
    (N'張小華', DATEADD(DAY, 2467, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '7367753@example.com', 'password3', 11, 'N'),
    (N'林美玲', DATEADD(DAY, 852, '1980-01-01'), 'F', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '1954184@example.com', 'password4', 234, 'N'),
    (N'陳志強', DATEADD(DAY, 445, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '1535755@example.com', 'password5', 565, 'N'),
    (N'吳依婷', DATEADD(DAY, 335, '1980-01-01'), 'F', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '5581596@example.com', 'password6', 704, 'N'),
    (N'劉建國', DATEADD(DAY, 1970, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '2914837@example.com', 'password7', 378, 'N'),
    (N'黃淑芬', DATEADD(DAY, 739, '1980-01-01'), 'F', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '4313528@example.com', 'password8', 414, 'N'),
    (N'何俊杰', DATEADD(DAY, 2671, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '926809@example.com', 'password9', 271, 'N'),
    (N'楊晴雯', DATEADD(DAY, 1796, '1980-01-01'), 'F', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '54283910@example.com', 'password10', 780, 'N'),
    (N'許慧娟', DATEADD(DAY, 1826, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '77357111@example.com', 'password11', 660, 'N'),
    (N'鄭文龍', DATEADD(DAY, 3000, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '65914512@example.com', 'password12', 839, 'N'),
    (N'謝宗翰', DATEADD(DAY, 236, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '13475213@example.com', 'password13', 203, 'N'),
    (N'邱雅琪', DATEADD(DAY, 939, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '41973814@example.com', 'password14', 547, 'N'),
    (N'郭子豪', DATEADD(DAY, 94, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '1764615@example.com', 'password15', 665, 'N'),
    (N'洪佩珊', DATEADD(DAY, 3144, '1980-01-01'), 'F', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '46294216@example.com', 'password16', 488, 'N'),
    (N'徐嘉文', DATEADD(DAY, 3580, '1980-01-01'), 'F', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '64259117@example.com', 'password17', 434, 'N'),
    (N'高文婷', DATEADD(DAY, 2127, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '40915918@example.com', 'password18', 682, 'N'),
    (N'周志豪', DATEADD(DAY, 646, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '8354619@example.com', 'password19', 770, 'N'),
    (N'曾麗芬', DATEADD(DAY, 1538, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '66995220@example.com', 'password20', 128, 'N'),
    (N'呂宏毅', DATEADD(DAY, 396, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '70398921@example.com', 'password21', 526, 'N'),
    (N'丁佳麗', DATEADD(DAY, 1089, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '69399922@example.com', 'password22', 386, 'N'),
    (N'潘瑞峰', DATEADD(DAY, 2114, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '4909223@example.com', 'password23', 430, 'N'),
    (N'姚美雲', DATEADD(DAY, 1550, '1980-01-01'), 'F', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '2544224@example.com', 'password24', 196, 'N'),
    (N'方國偉', DATEADD(DAY, 854, '1980-01-01'), 'F', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '41835325@example.com', 'password25', 234, 'N'),
    (N'馮小曼', DATEADD(DAY, 2410, '1980-01-01'), 'F', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '42676426@example.com', 'password26', 845, 'N'),
    (N'崔淑芳', DATEADD(DAY, 2581, '1980-01-01'), 'F', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '28135927@example.com', 'password27', 840, 'N'),
    (N'鐘偉豪', DATEADD(DAY, 1238, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '12534628@example.com', 'password28', 225, 'N'),
    (N'簡依琳', DATEADD(DAY, 2173, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '77814729@example.com', 'password29', 366, 'N'),
    (N'蔡建中', DATEADD(DAY, 1202, '1980-01-01'), 'M', DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 3650), '2010-01-01'), '81141330@example.com', 'password30', 225, 'N');

--DELETE FROM Member -- 刪除 29~88，使 ID 從 28 繼續開始
--WHERE ID BETWEEN 29 AND 88;
--DBCC CHECKIDENT ('Member', RESEED, 28);