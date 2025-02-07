---- 暫時刪除外鍵約束
--ALTER TABLE Detail DROP CONSTRAINT FK__Detail__OrderID__5BAD9CC8;
---- 清空 Detail 資料表並重置 IDENTITY
--DELETE FROM Detail;
--DBCC CHECKIDENT ('Detail', RESEED, 0);

---- 清空 Ord 資料表並重置 IDENTITY
--DELETE FROM Ord;
--DBCC CHECKIDENT ('Ord', RESEED, 0);

---- 重新添加外鍵約束
--ALTER TABLE Detail
--ADD CONSTRAINT FK__Detail__OrderID__5BAD9CC8 FOREIGN KEY (OrderID) REFERENCES Ord(ID);


DECLARE @i INT = 1; -- 計數器
DECLARE @menuid INT;
DECLARE @itemprice DECIMAL(10, 2);
DECLARE @amount INT;
DECLARE @orderid INT;
DECLARE @memid INT;
DECLARE @j INT; -- 商品數量計數器

WHILE @i <= 200 -- 200 筆訂單
BEGIN
    -- 隨機選擇 MemID（假設有 1 ~ 58 的 MemID）
    SET @memid = FLOOR(RAND() * 58) + 1;

    -- 插入資料到 Ord 資料表
    INSERT INTO Ord (MemID, OrderDate, Total)
    VALUES (@memid, DATEADD(DAY, ABS(CHECKSUM(NEWID()) % 365), '2019-01-01'), 0); -- 初始 Total 設為 0

    -- 獲取剛剛插入的 Ord 資料表的 ID
    SET @orderid = SCOPE_IDENTITY();

    -- 為每個訂單插入多個商品
    SET @j = 1;
    WHILE @j <= FLOOR(RAND() * 3) + 1 -- 每個訂單包含 1 ~ 3 個商品
    BEGIN
        -- 隨機選擇 MenuID（1 ~ 46）
        SET @menuid = FLOOR(RAND() * 46) + 1;

        -- 從 Menu 資料表中讀取該 MenuID 的價格
        SELECT @itemprice = Price FROM Menu WHERE ID = @menuid;

        -- 隨機生成 Amount（1 ~ 2）
        SET @amount = FLOOR(RAND() * 2) + 1;

        -- 插入資料到 Detail 資料表
        INSERT INTO Detail (OrderID, MenuID, ItemPrice, Amount)
        VALUES (@orderid, @menuid, @itemprice, @amount);

        -- 計數器遞增
        SET @j = @j + 1;
    END;

    -- 計算總金額並更新 Ord 表
    UPDATE Ord
    SET Total = (SELECT SUM(ItemPrice * Amount) FROM Detail WHERE OrderID = @orderid)
    WHERE ID = @orderid;

    -- 訂單計數器遞增
    SET @i = @i + 1;
END;

