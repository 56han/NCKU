using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SqlClient;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Linq;

namespace starbucks
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        //string cnstr = @"Data Source=(LocalDB)\MSSQLLocalDB;" +
        //                "AttachDbFilename=|DataDirectory|MyDB.mdf;" +
        //                "Integrated Security=True";

        string cnstr = @"Data Source=(LocalDB)\MSSQLLocalDB;" +
                        "AttachDbFilename=D:\\wanyihan\\碩二上\\C#\\final_project\\starbucks\\MyDB.mdf;" +
                        "Integrated Security=True";
        private void Form1_Load(object sender, EventArgs e)
        {
            using (SqlConnection cn = new SqlConnection())
            {
                yearrbtn.Checked = true;
                
                yearcomBox.Enabled = true;
                year2comBox.Enabled = false;
                monthcomBox.Enabled = false;
                startdateTP.Enabled = false;
                enddateTP.Enabled = false;

                reportrTB.Font = new Font("Consolas", 12); // 或 "Courier New"
                VIPrTB.Font = new Font("Consolas", 12);

                for (int i = DateTime.Now.Year - 5; i <= DateTime.Now.Year; i++)
                {
                    yearcomBox.Items.Add($"{i}");
                    year2comBox.Items.Add($"{i}");
                }
                yearcomBox.SelectedItem = "2024";

                for (int i = 1; i <= 12; i++)
                {
                    monthcomBox.Items.Add($"{i}");
                }

                staffStatecomBox.Items.Add("新進店員");
                staffStatecomBox.Items.Add("已離職");
                sexcomBox.Items.Add("男");
                sexcomBox.Items.Add("女");
                passwordtextBox.PasswordChar = '*';

                enddateTP.MaxDate = DateTime.Today; // 設置結束日期的最大值為今天
                enddateTP.MaxDate = DateTime.Today;
                birthdateTP.MaxDate = DateTime.Today;

                startdateTP.Value = DateTime.Today.AddDays(-7); // 預設為 7 天前
                enddateTP.Value = DateTime.Today;
                birthdateTP.Value = DateTime.Today.AddYears(-18); // 預設為 18 歲

                cn.ConnectionString = cnstr;

                string reportquery = @"
                    SELECT Ord.ID, Ord.MemID, Member.MemName, Ord.OrderDate, Ord.Total
                    FROM Ord 
                    INNER JOIN Member ON Ord.MemID = Member.ID 
                    ORDER BY Ord.ID DESC";

                DataSet ds = new DataSet();

                SqlDataAdapter daReport = new SqlDataAdapter(reportquery, cn);
                
                daReport.Fill(ds, "report");
                
                try
                {
                    StringBuilder reportsb = new StringBuilder();
                    reportsb.AppendLine("ID\tMemID\t\tMemName\tOrderDate\t\tTotal");
                    reportsb.AppendLine("--------------------------------------------------");

                    foreach (DataRow row in ds.Tables["report"].Rows)
                    {
                        string formattedDate = Convert.ToDateTime(row["OrderDate"]).ToString("yyyy-MM-dd");
                        reportsb.AppendLine($"{row["ID"]}\t{row["MemID"]}\t\t{row["MemName"]}\t{formattedDate}\t\t{row["Total"]}");
                    }

                    reportrTB.Text = reportsb.ToString();

                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            }
        }

        private void yearrbtn_CheckedChanged(object sender, EventArgs e)
        {
            yearcomBox.Enabled = true;
            year2comBox.Enabled = false;
            monthcomBox.Enabled = false;
            startdateTP.Enabled = false;
            enddateTP.Enabled = false;
        }

        private void monthrbtn_CheckedChanged(object sender, EventArgs e)
        {
            yearcomBox.Enabled = false;
            year2comBox.Enabled = true;
            monthcomBox.Enabled = true;
            startdateTP.Enabled = false;
            enddateTP.Enabled = false;
        }

        private void customrbtn_CheckedChanged(object sender, EventArgs e)
        {
            yearcomBox.Enabled = false;
            year2comBox.Enabled = false;
            monthcomBox.Enabled = false;
            startdateTP.Enabled = true;
            enddateTP.Enabled = true;
        }

        private void searchbtn_Click(object sender, EventArgs e)
        {
            using (SqlConnection cn = new SqlConnection())
            {
                cn.ConnectionString = cnstr;

                if (yearrbtn.Checked)
                {
                    if (yearcomBox.SelectedItem == null)
                    {
                        MessageBox.Show("請選擇年份！");
                        return;
                    }

                    string selectedYear = yearcomBox.SelectedItem.ToString();
                    lblshow.Text = $"以下是 {selectedYear} 年之報表";

                    string query = @"
                        SELECT Ord.ID, Ord.MemID, Member.MemName, Ord.OrderDate, Ord.Total
                        FROM Ord
                        INNER JOIN Member ON Ord.MemID = Member.ID
                        WHERE YEAR(Ord.OrderDate) = @SelectedYear
                        ORDER BY Ord.ID DESC";

                    SqlDataAdapter daOrder = CreateDataAdapter(query, cn, ("@SelectedYear", selectedYear));
                    ProcessYearlyReport(cn, daOrder, selectedYear);
                }
                else if (monthrbtn.Checked)
                {
                    if (year2comBox.SelectedItem == null || monthcomBox.SelectedItem == null)
                    {
                        MessageBox.Show("請選擇年份或月份！");
                        return;
                    }

                    string selectedYear = year2comBox.SelectedItem.ToString();
                    string selectedMonth = monthcomBox.SelectedItem.ToString();
                    lblshow.Text = $"以下是 {selectedYear} 年 {selectedMonth} 月之報表";

                    string query = @"
                        SELECT Ord.ID, Ord.MemID, Member.MemName, Ord.OrderDate, Ord.Total
                        FROM Ord
                        INNER JOIN Member ON Ord.MemID = Member.ID
                        WHERE YEAR(Ord.OrderDate) = @SelectedYear 
                        AND MONTH(Ord.OrderDate) = @SelectedMonth
                        ORDER BY Ord.ID DESC";

                    SqlDataAdapter daOrder = CreateDataAdapter(query, cn, ("@SelectedYear", selectedYear), ("@SelectedMonth", selectedMonth));
                    ProcessMonthlyReport(cn, daOrder, selectedYear, selectedMonth);
                }
                else if (customrbtn.Checked)
                {
                    string startDate = startdateTP.Value.ToString("yyyy-MM-dd");
                    string endDate = enddateTP.Value.ToString("yyyy-MM-dd");
                    lblshow.Text = $"以下是從 {startDate} 到 {endDate} 之報表";

                    string query = @"
                        SELECT Ord.ID, Ord.MemID, Member.MemName, Ord.OrderDate, Ord.Total
                        FROM Ord
                        INNER JOIN Member ON Ord.MemID = Member.ID
                        WHERE Ord.OrderDate BETWEEN @StartDate AND @EndDate
                        ORDER BY Ord.ID DESC";

                    SqlDataAdapter daOrder = CreateDataAdapter(query, cn, ("@StartDate", startDate), ("@EndDate", endDate));
                    ProcessCustomReport(cn, daOrder, startDate, endDate);
                }
                else
                {
                    MessageBox.Show("請先選擇欲查之時段");
                }
            }
        }

        private SqlDataAdapter CreateDataAdapter(string query, SqlConnection cn, params (string, object)[] parameters)
        {
            SqlDataAdapter adapter = new SqlDataAdapter(query, cn);
            foreach (var param in parameters)
            {
                adapter.SelectCommand.Parameters.AddWithValue(param.Item1, param.Item2);
            }
            return adapter;
        }

        private void ProcessYearlyReport(SqlConnection cn, SqlDataAdapter daOrder, string selectedYear)
        {
            DataSet ds = new DataSet();
            try
            {
                daOrder.Fill(ds, "order");

                if (ds.Tables["order"].Rows.Count > 0)
                {
                    int totalSum = 0;
                    int[] monthlySums = new int[12];
                    StringBuilder sb = new StringBuilder();

                    foreach (DataRow row in ds.Tables["order"].Rows)
                    {
                        DateTime orderDate = Convert.ToDateTime(row["OrderDate"]);
                        int total = Convert.ToInt32(row["Total"]);
                        totalSum += total;

                        int month = orderDate.Month;
                        monthlySums[month - 1] += total;
                    }

                    sb.AppendLine($"{selectedYear} 年每月消費金額：");
                    sb.AppendLine("日期\t\t\t\t金額");
                    sb.AppendLine("----------------------------------------");
                    for (int i = 0; i < 12; i++)
                    {
                        // 去掉 NT$，只保留數字，為了讓 "NT$" 對齊
                        string amount = $"{monthlySums[i]:N2}";

                        // 將年份、月份和 "NT$" 對齊，金額靠右
                        sb.AppendLine($"{selectedYear} 年  {(i + 1).ToString("D2")} 月 NT${amount,10}");
                    }

                    sb.AppendLine("\n----------------------------------------");
                    sb.AppendLine($"{selectedYear} 年度總營業金額：{totalSum:C}");

                    reportrTB.Text = sb.ToString();
                }
                else
                {
                    reportrTB.Text = "該年份無記錄。";
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"發生錯誤：{ex.Message}");
            }
        }

        private void ProcessMonthlyReport(SqlConnection cn, SqlDataAdapter daOrder, string selectedYear, string selectedMonth)
        {
            DataSet ds = new DataSet();
            try
            {
                daOrder.Fill(ds, "order");

                int daysInMonth = DateTime.DaysInMonth(Convert.ToInt32(selectedYear), Convert.ToInt32(selectedMonth));
                Dictionary<DateTime, int> dailySums = new Dictionary<DateTime, int>();
                StringBuilder sb = new StringBuilder();

                DateTime start = new DateTime(Convert.ToInt32(selectedYear), Convert.ToInt32(selectedMonth), 1);
                for (int i = 0; i < daysInMonth; i++)
                {
                    dailySums[start.AddDays(i)] = 0;
                }

                string title = $"{selectedYear} 年 {selectedMonth} 月每日消費金額：";
                string totalTitle = $"{selectedYear} 年 {selectedMonth} 月總營業金額：{{totalSum}}";

                GenerateDailyReport(ds, dailySums, sb, title, totalTitle);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"發生錯誤：{ex.Message}");
            }
        }

        private void ProcessCustomReport(SqlConnection cn, SqlDataAdapter daOrder, string startDate, string endDate)
        {
            DataSet ds = new DataSet();
            try
            {
                daOrder.Fill(ds, "order");

                DateTime start = DateTime.Parse(startDate);
                DateTime end = DateTime.Parse(endDate);
                int daysNum = (end - start).Days + 1;
                Dictionary<DateTime, int> dailySums = new Dictionary<DateTime, int>();
                StringBuilder sb = new StringBuilder();

                for (int i = 0; i < daysNum; i++)
                {
                    dailySums[start.AddDays(i)] = 0;
                }

                string title = $"從 {startDate} 到 {endDate} 每日消費金額：";
                string totalTitle = $"從 {startDate} 到 {endDate} 總營業金額：{{totalSum}}";

                GenerateDailyReport(ds, dailySums, sb, title, totalTitle);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"發生錯誤：{ex.Message}");
            }
        }

        private void GenerateDailyReport(DataSet ds, Dictionary<DateTime, int> dailySums, StringBuilder sb, string title, string totalTitle)
        {
            reportrTB.Clear();
            reportrTB.Font = new Font("Consolas", 12);

            if (ds.Tables["order"].Rows.Count > 0)
            {
                int totalSum = 0;

                // 累加每日金額
                foreach (DataRow row in ds.Tables["order"].Rows)
                {
                    DateTime orderDate = Convert.ToDateTime(row["OrderDate"]);
                    int total = Convert.ToInt32(row["Total"]);
                    totalSum += total;

                    if (dailySums.ContainsKey(orderDate))
                    {
                        dailySums[orderDate] += total;
                    }
                }

                sb.AppendLine(title);
                sb.AppendLine("日期\t\t\t\t金額");
                sb.AppendLine("----------------------------------------");

                foreach (var day in dailySums)
                {
                    sb.AppendLine($"{day.Key:yyyy-MM-dd}   NT$ {day.Value,8:N2}");
                }

                sb.AppendLine("\n----------------------------------------");
                sb.AppendLine(totalTitle.Replace("{totalSum}", totalSum.ToString("C")));

                reportrTB.Text = sb.ToString();
                reportrTB.SelectAll();
                reportrTB.SelectionFont = new Font("Consolas", 12);
            }
            else
            {
                reportrTB.Text = "沒有符合條件的記錄。";
            }
        }


        private void startdateTP_ValueChanged(object sender, EventArgs e)
        {
            // 動態調整 enddateTP 的最小值
            enddateTP.MinDate = startdateTP.Value;

            if (startdateTP.Value > enddateTP.Value)
            {
                enddateTP.Value = startdateTP.Value; // 調整結束日期
            }
        }

        private void enddateTP_ValueChanged(object sender, EventArgs e)
        {
            // 動態調整 startdateTP 的最大值
            startdateTP.MaxDate = enddateTP.Value;

            if (enddateTP.Value < startdateTP.Value)
            {
                startdateTP.Value = enddateTP.Value; // 調整開始日期
            }
        }

        private void yesvipbtn_Click(object sender, EventArgs e)
        {
            if(namesearchtextBox.Text == "")
            {
                MessageBox.Show("請輸入姓名！");
            }

            if (staffStatecomBox.SelectedItem == null)
            {
                MessageBox.Show("請選擇店員狀態！");
                return;
            }

            // 獲取使用者選擇的狀態
            string selectedState = staffStatecomBox.SelectedItem.ToString();
            char staffState = selectedState == "新進店員" ? 'Y' : 'N';

            string query = "SELECT * FROM Member WHERE MemName = @MemName";

            using (SqlConnection cn = new SqlConnection())
            {
                try
                {
                    cn.ConnectionString = cnstr;
                    cn.Open();

                    SqlDataAdapter adapter = new SqlDataAdapter(query, cn);
                    adapter.SelectCommand.Parameters.AddWithValue("@MemName", namesearchtextBox.Text);

                    DataTable memberTable = new DataTable();
                    adapter.Fill(memberTable);

                    // 使用 LINQ 查找目標資料行
                    var targetRow = memberTable.AsEnumerable().FirstOrDefault(row => row.Field<string>("MemName") == namesearchtextBox.Text);

                    if (targetRow != null)
                    {
                        targetRow["Staff"] = staffState;

                        SqlCommandBuilder commandBuilder = new SqlCommandBuilder(adapter);
                        adapter.UpdateCommand = commandBuilder.GetUpdateCommand();

                        // 提交更新到資料庫
                        adapter.Update(memberTable);

                        MessageBox.Show($"成功更新 {namesearchtextBox.Text} 的 Staff 狀態為 '{staffState}'！");
                    }
                    else
                    {
                        MessageBox.Show($"找不到名為 {namesearchtextBox.Text} 的會員！");
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"發生錯誤：{ex.Message}");
                }
            }
        }

        private void notvipbtn_Click(object sender, EventArgs e)
        {
            if (nametextBox.Text == "" || sexcomBox.SelectedItem == null || emailtextBox.Text == "" || passwordtextBox.Text == "")
            {
                MessageBox.Show("請填寫所有必填欄位！");
            }

            string selectQuery = "SELECT COUNT(*) FROM Member WHERE MemName = @MemName";
            string insertQuery = @"
                INSERT INTO Member (MemName, Birth, Sex, JoinDate, Email, MemPassword, Points, Staff)
                VALUES (@MemName, @Birth, @Sex, @JoinDate, @Email, @MemPassword, @Points, @Staff)";

            string selectedSex = sexcomBox.SelectedItem.ToString();
            char sexstate = selectedSex == "女" ? 'F' : 'M';

            using (SqlConnection cn = new SqlConnection())
            {
                try
                {
                    cn.ConnectionString = cnstr;
                    cn.Open();

                    // 查詢是否已存在相同的 MemName
                    SqlCommand selectCmd = new SqlCommand(selectQuery, cn);
                    selectCmd.Parameters.AddWithValue("@MemName", nametextBox.Text);
                    int count = (int)selectCmd.ExecuteScalar(); // 獲取查詢結果

                    if (count > 0)
                    {
                        MessageBox.Show($"會員 {nametextBox.Text} 已存在，無需新增！");
                        return;
                    }

                    // 如果不存在，執行插入操作
                    SqlCommand insertCmd = new SqlCommand(insertQuery, cn);
                    insertCmd.Parameters.AddWithValue("@MemName", nametextBox.Text);
                    insertCmd.Parameters.AddWithValue("@Birth", birthdateTP.Value.Date);        // 只取日期
                    insertCmd.Parameters.AddWithValue("@Sex", sexstate);
                    insertCmd.Parameters.AddWithValue("@JoinDate", DateTime.Now);               // 今天日期
                    insertCmd.Parameters.AddWithValue("@Email", emailtextBox.Text);
                    insertCmd.Parameters.AddWithValue("@MemPassword", passwordtextBox.Text);
                    insertCmd.Parameters.AddWithValue("@Points", 0);                            // 預設 Points 為 0
                    insertCmd.Parameters.AddWithValue("@Staff", 'Y');                           // 預設 Staff 為 'Y'

                    int rowsAffected = insertCmd.ExecuteNonQuery();

                    if (rowsAffected > 0)
                    {
                        MessageBox.Show("人員新增成功！");
                    }
                    else
                    {
                        MessageBox.Show("新增失敗，請檢查輸入資料。");
                    }

                }
                catch (Exception ex)
                {
                    MessageBox.Show($"發生錯誤：{ex.Message}");
                }
            }
        }

        private void tabControl1_SelectedIndexChanged(object sender, EventArgs e)
        {
            using (SqlConnection cn = new SqlConnection())
            {
                if (tabControl1.SelectedTab == VIPTabPage)
                {
                    string vipquery = @"
                        SELECT ID, MemName, Birth, Sex, Points, Staff
                        FROM Member 
                        ORDER BY Member.ID";

                    try
                    {
                        cn.ConnectionString = cnstr;
                        DataSet ds = new DataSet();
                        SqlDataAdapter daVIP = new SqlDataAdapter(vipquery, cn);

                        daVIP.Fill(ds, "VIP");

                        StringBuilder vipsb = new StringBuilder();
                        vipsb.AppendLine("ID\tMemName\tBirth\t\t\tSex\tPoints\tStaff");
                        vipsb.AppendLine("--------------------------------------------------------");

                        foreach (DataRow row in ds.Tables["VIP"].Rows)
                        {
                            string formattedDate = Convert.ToDateTime(row["Birth"]).ToString("yyyy-MM-dd");
                            vipsb.AppendLine($"{row["ID"]}\t{row["MemName"]}\t{formattedDate}\t\t{row["Sex"]}\t{row["Points"]}\t\t{row["Staff"]}");
                        }

                        VIPrTB.Text = vipsb.ToString();
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"發生錯誤：{ex.Message}");
                    }

                }
                else if (tabControl1.SelectedTab == ReportTabPage)
                {
                    string reportquery = @"
                        SELECT Ord.ID, Ord.MemID, Member.MemName, Ord.OrderDate, Ord.Total
                        FROM Ord 
                        INNER JOIN Member ON Ord.MemID = Member.ID 
                        ORDER BY Ord.ID DESC";

                    try
                    {
                        cn.ConnectionString = cnstr;
                        DataSet ds = new DataSet();
                        SqlDataAdapter daReport = new SqlDataAdapter(reportquery, cn);

                        daReport.Fill(ds, "report");

                        StringBuilder reportsb = new StringBuilder();
                        reportsb.AppendLine("ID\tMemID\t\tMemName\tOrderDate\t\tTotal");
                        reportsb.AppendLine("--------------------------------------------------");

                        foreach (DataRow row in ds.Tables["report"].Rows)
                        {
                            string formattedDate = Convert.ToDateTime(row["OrderDate"]).ToString("yyyy-MM-dd");
                            reportsb.AppendLine($"{row["ID"]}\t{row["MemID"]}\t\t{row["MemName"]}\t{formattedDate}\t\t{row["Total"]}");
                        }

                        reportrTB.Text = reportsb.ToString();

                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"發生錯誤：{ex.Message}");
                    }
                }
            }
        }

        
    }
}
