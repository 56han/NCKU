namespace starbucks
{
    partial class Form1
    {
        /// <summary>
        /// 設計工具所需的變數。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清除任何使用中的資源。
        /// </summary>
        /// <param name="disposing">如果應該處置受控資源則為 true，否則為 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form 設計工具產生的程式碼

        /// <summary>
        /// 此為設計工具支援所需的方法 - 請勿使用程式碼編輯器修改
        /// 這個方法的內容。
        /// </summary>
        private void InitializeComponent()
        {
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.ReportTabPage = new System.Windows.Forms.TabPage();
            this.customrbtn = new System.Windows.Forms.RadioButton();
            this.lblshow = new System.Windows.Forms.Label();
            this.searchbtn = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.monthcomBox = new System.Windows.Forms.ComboBox();
            this.year2comBox = new System.Windows.Forms.ComboBox();
            this.yearcomBox = new System.Windows.Forms.ComboBox();
            this.reportrTB = new System.Windows.Forms.RichTextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.enddateTP = new System.Windows.Forms.DateTimePicker();
            this.startdateTP = new System.Windows.Forms.DateTimePicker();
            this.label1 = new System.Windows.Forms.Label();
            this.monthrbtn = new System.Windows.Forms.RadioButton();
            this.yearrbtn = new System.Windows.Forms.RadioButton();
            this.VIPTabPage = new System.Windows.Forms.TabPage();
            this.VIPrTB = new System.Windows.Forms.RichTextBox();
            this.NewStafftabPage = new System.Windows.Forms.TabPage();
            this.tabControl2 = new System.Windows.Forms.TabControl();
            this.yesviptabPage = new System.Windows.Forms.TabPage();
            this.yesvipbtn = new System.Windows.Forms.Button();
            this.label11 = new System.Windows.Forms.Label();
            this.staffStatecomBox = new System.Windows.Forms.ComboBox();
            this.namesearchtextBox = new System.Windows.Forms.TextBox();
            this.label12 = new System.Windows.Forms.Label();
            this.notviptabPage = new System.Windows.Forms.TabPage();
            this.notvipbtn = new System.Windows.Forms.Button();
            this.emailtextBox = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.birthdateTP = new System.Windows.Forms.DateTimePicker();
            this.label9 = new System.Windows.Forms.Label();
            this.sexcomBox = new System.Windows.Forms.ComboBox();
            this.label10 = new System.Windows.Forms.Label();
            this.passwordtextBox = new System.Windows.Forms.TextBox();
            this.nametextBox = new System.Windows.Forms.TextBox();
            this.tabControl1.SuspendLayout();
            this.ReportTabPage.SuspendLayout();
            this.VIPTabPage.SuspendLayout();
            this.NewStafftabPage.SuspendLayout();
            this.tabControl2.SuspendLayout();
            this.yesviptabPage.SuspendLayout();
            this.notviptabPage.SuspendLayout();
            this.SuspendLayout();
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.ReportTabPage);
            this.tabControl1.Controls.Add(this.VIPTabPage);
            this.tabControl1.Controls.Add(this.NewStafftabPage);
            this.tabControl1.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.tabControl1.ItemSize = new System.Drawing.Size(80, 30);
            this.tabControl1.Location = new System.Drawing.Point(12, 12);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(1247, 775);
            this.tabControl1.TabIndex = 0;
            this.tabControl1.SelectedIndexChanged += new System.EventHandler(this.tabControl1_SelectedIndexChanged);
            // 
            // ReportTabPage
            // 
            this.ReportTabPage.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.ReportTabPage.Controls.Add(this.customrbtn);
            this.ReportTabPage.Controls.Add(this.lblshow);
            this.ReportTabPage.Controls.Add(this.searchbtn);
            this.ReportTabPage.Controls.Add(this.label5);
            this.ReportTabPage.Controls.Add(this.label4);
            this.ReportTabPage.Controls.Add(this.label3);
            this.ReportTabPage.Controls.Add(this.monthcomBox);
            this.ReportTabPage.Controls.Add(this.year2comBox);
            this.ReportTabPage.Controls.Add(this.yearcomBox);
            this.ReportTabPage.Controls.Add(this.reportrTB);
            this.ReportTabPage.Controls.Add(this.label2);
            this.ReportTabPage.Controls.Add(this.enddateTP);
            this.ReportTabPage.Controls.Add(this.startdateTP);
            this.ReportTabPage.Controls.Add(this.label1);
            this.ReportTabPage.Controls.Add(this.monthrbtn);
            this.ReportTabPage.Controls.Add(this.yearrbtn);
            this.ReportTabPage.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.ReportTabPage.Location = new System.Drawing.Point(4, 34);
            this.ReportTabPage.Name = "ReportTabPage";
            this.ReportTabPage.Padding = new System.Windows.Forms.Padding(3);
            this.ReportTabPage.Size = new System.Drawing.Size(1239, 737);
            this.ReportTabPage.TabIndex = 0;
            this.ReportTabPage.Text = "報表";
            this.ReportTabPage.UseVisualStyleBackColor = true;
            // 
            // customrbtn
            // 
            this.customrbtn.AutoSize = true;
            this.customrbtn.Location = new System.Drawing.Point(652, 60);
            this.customrbtn.Name = "customrbtn";
            this.customrbtn.Size = new System.Drawing.Size(73, 29);
            this.customrbtn.TabIndex = 17;
            this.customrbtn.Text = "自訂";
            this.customrbtn.UseVisualStyleBackColor = true;
            this.customrbtn.CheckedChanged += new System.EventHandler(this.customrbtn_CheckedChanged);
            // 
            // lblshow
            // 
            this.lblshow.AutoSize = true;
            this.lblshow.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.lblshow.Location = new System.Drawing.Point(21, 21);
            this.lblshow.Name = "lblshow";
            this.lblshow.Size = new System.Drawing.Size(212, 25);
            this.lblshow.TabIndex = 16;
            this.lblshow.Text = "請選擇欲查詢之時段！";
            // 
            // searchbtn
            // 
            this.searchbtn.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.searchbtn.Location = new System.Drawing.Point(1056, 98);
            this.searchbtn.Name = "searchbtn";
            this.searchbtn.Size = new System.Drawing.Size(151, 38);
            this.searchbtn.TabIndex = 15;
            this.searchbtn.Text = "查詢";
            this.searchbtn.UseVisualStyleBackColor = true;
            this.searchbtn.Click += new System.EventHandler(this.searchbtn_Click);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(153, 107);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(32, 25);
            this.label5.TabIndex = 14;
            this.label5.Text = "年";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(561, 107);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(32, 25);
            this.label4.TabIndex = 13;
            this.label4.Text = "月";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(396, 107);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(32, 25);
            this.label3.TabIndex = 12;
            this.label3.Text = "年";
            // 
            // monthcomBox
            // 
            this.monthcomBox.FormattingEnabled = true;
            this.monthcomBox.Location = new System.Drawing.Point(434, 103);
            this.monthcomBox.Name = "monthcomBox";
            this.monthcomBox.Size = new System.Drawing.Size(121, 33);
            this.monthcomBox.TabIndex = 11;
            // 
            // year2comBox
            // 
            this.year2comBox.FormattingEnabled = true;
            this.year2comBox.Location = new System.Drawing.Point(269, 103);
            this.year2comBox.Name = "year2comBox";
            this.year2comBox.Size = new System.Drawing.Size(121, 33);
            this.year2comBox.TabIndex = 10;
            // 
            // yearcomBox
            // 
            this.yearcomBox.FormattingEnabled = true;
            this.yearcomBox.Location = new System.Drawing.Point(26, 103);
            this.yearcomBox.Name = "yearcomBox";
            this.yearcomBox.Size = new System.Drawing.Size(121, 33);
            this.yearcomBox.TabIndex = 9;
            // 
            // reportrTB
            // 
            this.reportrTB.Location = new System.Drawing.Point(26, 153);
            this.reportrTB.Name = "reportrTB";
            this.reportrTB.Size = new System.Drawing.Size(1181, 564);
            this.reportrTB.TabIndex = 8;
            this.reportrTB.Text = "";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(730, 103);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(52, 25);
            this.label2.TabIndex = 7;
            this.label2.Text = "結束";
            // 
            // enddateTP
            // 
            this.enddateTP.Location = new System.Drawing.Point(788, 101);
            this.enddateTP.Name = "enddateTP";
            this.enddateTP.Size = new System.Drawing.Size(200, 34);
            this.enddateTP.TabIndex = 6;
            this.enddateTP.ValueChanged += new System.EventHandler(this.enddateTP_ValueChanged);
            // 
            // startdateTP
            // 
            this.startdateTP.Location = new System.Drawing.Point(788, 55);
            this.startdateTP.Name = "startdateTP";
            this.startdateTP.Size = new System.Drawing.Size(200, 34);
            this.startdateTP.TabIndex = 5;
            this.startdateTP.ValueChanged += new System.EventHandler(this.startdateTP_ValueChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(730, 62);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(52, 25);
            this.label1.TabIndex = 4;
            this.label1.Text = "開始";
            // 
            // monthrbtn
            // 
            this.monthrbtn.AutoSize = true;
            this.monthrbtn.Location = new System.Drawing.Point(269, 60);
            this.monthrbtn.Name = "monthrbtn";
            this.monthrbtn.Size = new System.Drawing.Size(53, 29);
            this.monthrbtn.TabIndex = 1;
            this.monthrbtn.Text = "月";
            this.monthrbtn.UseVisualStyleBackColor = true;
            this.monthrbtn.CheckedChanged += new System.EventHandler(this.monthrbtn_CheckedChanged);
            // 
            // yearrbtn
            // 
            this.yearrbtn.AutoSize = true;
            this.yearrbtn.Checked = true;
            this.yearrbtn.Location = new System.Drawing.Point(26, 60);
            this.yearrbtn.Name = "yearrbtn";
            this.yearrbtn.Size = new System.Drawing.Size(53, 29);
            this.yearrbtn.TabIndex = 0;
            this.yearrbtn.TabStop = true;
            this.yearrbtn.Text = "年";
            this.yearrbtn.UseVisualStyleBackColor = true;
            this.yearrbtn.CheckedChanged += new System.EventHandler(this.yearrbtn_CheckedChanged);
            // 
            // VIPTabPage
            // 
            this.VIPTabPage.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.VIPTabPage.Controls.Add(this.VIPrTB);
            this.VIPTabPage.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.VIPTabPage.Location = new System.Drawing.Point(4, 34);
            this.VIPTabPage.Name = "VIPTabPage";
            this.VIPTabPage.Padding = new System.Windows.Forms.Padding(3);
            this.VIPTabPage.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.VIPTabPage.Size = new System.Drawing.Size(1239, 737);
            this.VIPTabPage.TabIndex = 1;
            this.VIPTabPage.Text = "會員列表";
            this.VIPTabPage.UseVisualStyleBackColor = true;
            // 
            // VIPrTB
            // 
            this.VIPrTB.Location = new System.Drawing.Point(6, 6);
            this.VIPrTB.Name = "VIPrTB";
            this.VIPrTB.Size = new System.Drawing.Size(1223, 721);
            this.VIPrTB.TabIndex = 0;
            this.VIPrTB.Text = "";
            // 
            // NewStafftabPage
            // 
            this.NewStafftabPage.Controls.Add(this.tabControl2);
            this.NewStafftabPage.Location = new System.Drawing.Point(4, 34);
            this.NewStafftabPage.Name = "NewStafftabPage";
            this.NewStafftabPage.Size = new System.Drawing.Size(1239, 737);
            this.NewStafftabPage.TabIndex = 2;
            this.NewStafftabPage.Text = "修改管理者狀態";
            this.NewStafftabPage.UseVisualStyleBackColor = true;
            // 
            // tabControl2
            // 
            this.tabControl2.Controls.Add(this.yesviptabPage);
            this.tabControl2.Controls.Add(this.notviptabPage);
            this.tabControl2.Location = new System.Drawing.Point(3, 3);
            this.tabControl2.Name = "tabControl2";
            this.tabControl2.SelectedIndex = 0;
            this.tabControl2.Size = new System.Drawing.Size(1233, 731);
            this.tabControl2.TabIndex = 20;
            // 
            // yesviptabPage
            // 
            this.yesviptabPage.Controls.Add(this.yesvipbtn);
            this.yesviptabPage.Controls.Add(this.label11);
            this.yesviptabPage.Controls.Add(this.staffStatecomBox);
            this.yesviptabPage.Controls.Add(this.namesearchtextBox);
            this.yesviptabPage.Controls.Add(this.label12);
            this.yesviptabPage.Location = new System.Drawing.Point(4, 34);
            this.yesviptabPage.Name = "yesviptabPage";
            this.yesviptabPage.Padding = new System.Windows.Forms.Padding(3);
            this.yesviptabPage.Size = new System.Drawing.Size(1225, 693);
            this.yesviptabPage.TabIndex = 1;
            this.yesviptabPage.Text = "已註冊之會員";
            this.yesviptabPage.UseVisualStyleBackColor = true;
            // 
            // yesvipbtn
            // 
            this.yesvipbtn.Location = new System.Drawing.Point(261, 153);
            this.yesvipbtn.Name = "yesvipbtn";
            this.yesvipbtn.Size = new System.Drawing.Size(120, 35);
            this.yesvipbtn.TabIndex = 19;
            this.yesvipbtn.Text = "確認";
            this.yesvipbtn.UseVisualStyleBackColor = true;
            this.yesvipbtn.Click += new System.EventHandler(this.yesvipbtn_Click);
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.label11.Location = new System.Drawing.Point(26, 50);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(152, 25);
            this.label11.TabIndex = 15;
            this.label11.Text = "透過姓名查詢：";
            // 
            // staffStatecomBox
            // 
            this.staffStatecomBox.FormattingEnabled = true;
            this.staffStatecomBox.Location = new System.Drawing.Point(205, 93);
            this.staffStatecomBox.Name = "staffStatecomBox";
            this.staffStatecomBox.Size = new System.Drawing.Size(176, 33);
            this.staffStatecomBox.TabIndex = 18;
            // 
            // namesearchtextBox
            // 
            this.namesearchtextBox.Location = new System.Drawing.Point(184, 41);
            this.namesearchtextBox.Name = "namesearchtextBox";
            this.namesearchtextBox.Size = new System.Drawing.Size(197, 34);
            this.namesearchtextBox.TabIndex = 16;
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.label12.Location = new System.Drawing.Point(26, 101);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(172, 25);
            this.label12.TabIndex = 17;
            this.label12.Text = "修改管理者狀態：";
            // 
            // notviptabPage
            // 
            this.notviptabPage.Controls.Add(this.notvipbtn);
            this.notviptabPage.Controls.Add(this.emailtextBox);
            this.notviptabPage.Controls.Add(this.label6);
            this.notviptabPage.Controls.Add(this.label7);
            this.notviptabPage.Controls.Add(this.label8);
            this.notviptabPage.Controls.Add(this.birthdateTP);
            this.notviptabPage.Controls.Add(this.label9);
            this.notviptabPage.Controls.Add(this.sexcomBox);
            this.notviptabPage.Controls.Add(this.label10);
            this.notviptabPage.Controls.Add(this.passwordtextBox);
            this.notviptabPage.Controls.Add(this.nametextBox);
            this.notviptabPage.Location = new System.Drawing.Point(4, 34);
            this.notviptabPage.Name = "notviptabPage";
            this.notviptabPage.Padding = new System.Windows.Forms.Padding(3);
            this.notviptabPage.Size = new System.Drawing.Size(1225, 693);
            this.notviptabPage.TabIndex = 0;
            this.notviptabPage.Text = "尚未註冊之會員";
            this.notviptabPage.UseVisualStyleBackColor = true;
            // 
            // notvipbtn
            // 
            this.notvipbtn.Location = new System.Drawing.Point(194, 284);
            this.notvipbtn.Name = "notvipbtn";
            this.notvipbtn.Size = new System.Drawing.Size(120, 35);
            this.notvipbtn.TabIndex = 19;
            this.notvipbtn.Text = "確定";
            this.notvipbtn.UseVisualStyleBackColor = true;
            this.notvipbtn.Click += new System.EventHandler(this.notvipbtn_Click);
            // 
            // emailtextBox
            // 
            this.emailtextBox.Location = new System.Drawing.Point(117, 171);
            this.emailtextBox.Name = "emailtextBox";
            this.emailtextBox.Size = new System.Drawing.Size(197, 34);
            this.emailtextBox.TabIndex = 9;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.label6.Location = new System.Drawing.Point(39, 36);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(72, 25);
            this.label6.TabIndex = 0;
            this.label6.Text = "姓名：";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.label7.Location = new System.Drawing.Point(39, 85);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(72, 25);
            this.label7.TabIndex = 1;
            this.label7.Text = "生日：";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.label8.Location = new System.Drawing.Point(39, 132);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(72, 25);
            this.label8.TabIndex = 2;
            this.label8.Text = "性別：";
            // 
            // birthdateTP
            // 
            this.birthdateTP.Location = new System.Drawing.Point(117, 78);
            this.birthdateTP.Name = "birthdateTP";
            this.birthdateTP.Size = new System.Drawing.Size(197, 34);
            this.birthdateTP.TabIndex = 14;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.label9.Location = new System.Drawing.Point(39, 180);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(72, 25);
            this.label9.TabIndex = 3;
            this.label9.Text = "信箱：";
            // 
            // sexcomBox
            // 
            this.sexcomBox.FormattingEnabled = true;
            this.sexcomBox.Location = new System.Drawing.Point(117, 124);
            this.sexcomBox.Name = "sexcomBox";
            this.sexcomBox.Size = new System.Drawing.Size(197, 33);
            this.sexcomBox.TabIndex = 13;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Font = new System.Drawing.Font("微軟正黑體", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.label10.Location = new System.Drawing.Point(39, 229);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(72, 25);
            this.label10.TabIndex = 4;
            this.label10.Text = "密碼：";
            // 
            // passwordtextBox
            // 
            this.passwordtextBox.Location = new System.Drawing.Point(117, 220);
            this.passwordtextBox.Name = "passwordtextBox";
            this.passwordtextBox.Size = new System.Drawing.Size(197, 34);
            this.passwordtextBox.TabIndex = 10;
            // 
            // nametextBox
            // 
            this.nametextBox.Location = new System.Drawing.Point(117, 27);
            this.nametextBox.Name = "nametextBox";
            this.nametextBox.Size = new System.Drawing.Size(197, 34);
            this.nametextBox.TabIndex = 7;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1271, 799);
            this.Controls.Add(this.tabControl1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.tabControl1.ResumeLayout(false);
            this.ReportTabPage.ResumeLayout(false);
            this.ReportTabPage.PerformLayout();
            this.VIPTabPage.ResumeLayout(false);
            this.NewStafftabPage.ResumeLayout(false);
            this.tabControl2.ResumeLayout(false);
            this.yesviptabPage.ResumeLayout(false);
            this.yesviptabPage.PerformLayout();
            this.notviptabPage.ResumeLayout(false);
            this.notviptabPage.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage ReportTabPage;
        private System.Windows.Forms.TabPage VIPTabPage;
        private System.Windows.Forms.RadioButton monthrbtn;
        private System.Windows.Forms.RadioButton yearrbtn;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.DateTimePicker enddateTP;
        private System.Windows.Forms.DateTimePicker startdateTP;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.RichTextBox reportrTB;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.ComboBox monthcomBox;
        private System.Windows.Forms.ComboBox year2comBox;
        private System.Windows.Forms.ComboBox yearcomBox;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.RichTextBox VIPrTB;
        private System.Windows.Forms.Button searchbtn;
        private System.Windows.Forms.Label lblshow;
        private System.Windows.Forms.RadioButton customrbtn;
        private System.Windows.Forms.TabPage NewStafftabPage;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.ComboBox sexcomBox;
        private System.Windows.Forms.TextBox passwordtextBox;
        private System.Windows.Forms.TextBox emailtextBox;
        private System.Windows.Forms.TextBox nametextBox;
        private System.Windows.Forms.DateTimePicker birthdateTP;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.TextBox namesearchtextBox;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.ComboBox staffStatecomBox;
        private System.Windows.Forms.Button notvipbtn;
        private System.Windows.Forms.TabControl tabControl2;
        private System.Windows.Forms.TabPage notviptabPage;
        private System.Windows.Forms.TabPage yesviptabPage;
        private System.Windows.Forms.Button yesvipbtn;
    }
}

