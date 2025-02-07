using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using PicWeb.Models;

namespace PicWeb.Controllers
{
    public class AccountController : BaseController
    {
        //PicWebDBEntities db = new PicWebDBEntities();
        public ActionResult Login()
        {
            return View();
        }

        [HttpPost]
        public ActionResult Login(string username, string password)
        {
            // 檢查帳密是否正確
            var member = db.Member.FirstOrDefault(m => m.MemName == username && m.MemPassword == password);

            if (member != null)
            {
                // 登入成功，重定向到作品列表
                Session["MemberId"] = member.ID;
                Session["MemberName"] = member.MemName;
                return RedirectToAction("Index", "Post");
            }
            else
            {
                // 登入失敗，加入錯誤訊息
                TempData["ErrorMessage"] = "帳號或密碼錯誤，請重新輸入。";
                return View();
            }
        }

        public ActionResult Register()
        {
            return View();
        }

        [HttpPost]
        public ActionResult Register(string username, string password, string confirmPassword)
        {
            if (!IsValidUsername(username))
            {
                TempData["ErrorMessage"] = "帳號只能包含英文字母、數字和底線(_)。";
                return View();
            }

            if (password != confirmPassword)
            {
                TempData["ErrorMessage"] = "密碼與確認密碼不符，請重新輸入。";
                return View();
            }

            // 檢查帳號是否已存在
            if (db.Member.Any(m => m.MemName == username))
            {
                TempData["ErrorMessage"] = "此帳號已被使用，請選擇其他帳號。";
                return View();
            }

            // 建立新會員
            var newMember = new Member
            {
                MemName = username,
                MemPassword = password,
            };

            db.Member.Add(newMember);
            db.SaveChanges();

            TempData["SuccessMessage"] = "註冊成功！請登入。";
            return RedirectToAction("Login");
        }

        private bool IsValidUsername(string username)
        {
            // 只允許英文字母、數字和底線
            return !string.IsNullOrEmpty(username) &&
                   username.All(c => char.IsLetterOrDigit(c) || c == '_');
        }

        public ActionResult PersonProfile()
        {
            if (Session["MemberId"] == null)
            {
                return RedirectToAction("Login");
            }

            int memberId = (int)Session["MemberId"];
            var member = db.Member.Find(memberId);

            ViewBag.UserPosts = db.Post
                .Where(p => p.MemID == memberId)
                .ToList();

            return View(member);
        }

        [HttpGet]
        public ActionResult Edit()
        {
            if (Session["MemberId"] == null)
            {
                return RedirectToAction("Login");
            }

            int memberId = (int)Session["MemberId"];
            var member = db.Member.Find(memberId);
            return View(member);
        }

        [HttpPost]
        public ActionResult Edit(Member model)
        {
            if (ModelState.IsValid)
            {
                var member = db.Member.Find(model.ID);
                if (member != null)
                {
                    member.MemName = model.MemName;
                    member.MemIntro = model.MemIntro;
                    db.SaveChanges();

                    // 更新 Session 中的會員名稱
                    Session["MemberName"] = model.MemName;
                }
                return RedirectToAction("PersonProfile");
            }
            return View(model);
        }


        public ActionResult Logout()
        {
            Session.Clear();
            return RedirectToAction("Index", "Post");
        }
    }
}