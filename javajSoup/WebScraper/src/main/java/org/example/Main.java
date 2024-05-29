package org.example;

import org.json.JSONArray;
import org.json.JSONObject;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import javax.net.ssl.*;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;

public class Main {
    public static void main(String[] args) {

        try {

            String jsonContent = new String(Files.readAllBytes(Paths.get("new_data_68.json")));


            JSONArray jsonArray = new JSONArray(jsonContent);


            FileWriter writer = new FileWriter("veri.txt");
            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject jsonObject = jsonArray.getJSONObject(i);
                String text = jsonObject.getString("text");
                writer.write(text + "\n\n");
            }
            writer.close();

            System.out.println("JSON verisi başarıyla TXT dosyasına yazıldı.");
        } catch (IOException e) {
            e.printStackTrace();
        }

    /*    disableCertificateValidation();
        https://www.firat.edu.tr/tr/page/news?page=2
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter("duyuru.txt", true));

            for (int page = 35; page <= 40; page++) {
                String baseUrl = "https://www.firat.edu.tr/tr/page/news?page=" + page;
                Document mainPage = Jsoup.connect(baseUrl).get();
                Elements posts = mainPage.select(".item-content  a");
                //System.out.println("Posts verisi:" + posts);
                for (int i = 0; i < posts.size(); i += 2) {
                    Element post = posts.get(i);
                    String postUrl = post.absUrl("href");
                    if (!postUrl.isEmpty()) {
                        System.out.println("Duyuru URL'si: " + postUrl);
                        Document postPage = Jsoup.connect(postUrl).get();
                        Elements content = postPage.select(".post-content");

                        for (Element element : content) {
                            writer.write(element.text());
                            writer.newLine();
                        }

                        writer.newLine();
                    }
                }
            }

            writer.close();
            System.out.println("Tüm içerikler başarıyla duyuru.txt dosyasına yazıldı.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    } */

   /* private static void disableCertificateValidation() {
        try {
            TrustManager[] trustAllCerts = new TrustManager[]{
                    new X509TrustManager() {
                        public java.security.cert.X509Certificate[] getAcceptedIssuers() {
                            return null;
                        }

                        public void checkClientTrusted(
                                java.security.cert.X509Certificate[] certs, String authType) {
                        }

                        public void checkServerTrusted(
                                java.security.cert.X509Certificate[] certs, String authType) {
                        }
                    }
            };

            SSLContext sc = SSLContext.getInstance("TLS");
            sc.init(null, trustAllCerts, new java.security.SecureRandom());
            HttpsURLConnection.setDefaultSSLSocketFactory(sc.getSocketFactory());

            HostnameVerifier allHostsValid = (hostname, session) -> true;
            HttpsURLConnection.setDefaultHostnameVerifier(allHostsValid);
        } catch (NoSuchAlgorithmException | KeyManagementException e) {
            e.printStackTrace();
        } */
    }
}
