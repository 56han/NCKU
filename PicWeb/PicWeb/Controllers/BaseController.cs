using PicWeb.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace PicWeb.Controllers
{
    public class BaseController : Controller
    {
        // GET: Base
        protected PicWebDBEntities db = new PicWebDBEntities();

        protected override void OnActionExecuting(ActionExecutingContext filterContext)
        {
            base.OnActionExecuting(filterContext);
            // 取得所有相機並傳給 ViewBag
            ViewBag.AllCameras = db.Camera.ToList();
        }
    }
}