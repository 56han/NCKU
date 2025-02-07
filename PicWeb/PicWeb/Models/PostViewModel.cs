using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using PicWeb.Models;

namespace PicWeb.Models
{
    public class PostViewModel
    {
        public int Id { get; set; }
        public int CamId { get; set; }
        public int MemId { get; set; }
        public string Note { get; set; }
        public string Pic { get; set; }
        public int LikeNum { get; set; }
        public DateTime PostDate { get; set; }

        // 導航屬性
        public string CamName { get; set; }
        public string MemName { get; set; }

        // 如果需要存放完整的關聯物件
        public virtual Camera Camera { get; set; }
        public virtual Member Member { get; set; }

        public bool IsLiked { get; set; }
    }
}