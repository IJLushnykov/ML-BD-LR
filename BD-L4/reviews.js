db.createCollection("reviews", { 
    capped: true, 
    size: 1024, 
    max: 5 
});

db.reviews.insertMany([
    { "review_id": 1, "customer_name": "Andrii Rodinin", "rating": 5, "comment": "5 зірок!", "date": ISODate("2024-05-25") },
    { "review_id": 2, "customer_name": "Ivan Lushnykov", "rating": 4, "comment": "Гарна якість, але ціни кусаються!", "date": ISODate("2024-05-26") },
    { "review_id": 3, "customer_name": "Alla Mazur", "rating": 3, "comment": "Середнячок", "date": ISODate("2024-05-27") },
    { "review_id": 4, "customer_name": "Olxeii Nedilya", "rating": 2, "comment": "Якість не відповідає цінам.", "date": ISODate("2024-05-28") },
    { "review_id": 5, "customer_name": "Oleg Hrach", "rating": 1, "comment": "Все незадовільно.", "date": ISODate("2024-05-29") }
]);

db.reviews.find().pretty()

db.reviews.insertOne(
    { "review_id": 6, "customer_name": "Anna Antonenko", "rating": 5, "comment": "Клас!", "date": ISODate("2024-05-30") }
);

db.reviews.insertOne(
    { "review_id": 7, "customer_name": "Anton Rechik", "rating": 4, "comment": "4*", "date": ISODate("2024-05-31") }
);
db.reviews.find().pretty()
